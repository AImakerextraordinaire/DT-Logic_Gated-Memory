"""Phase 2: consolidation-gated writes + deterministic recall gate.

Each epoch is one *encounter* with every experience:
  1. PROBE    epistemic confidence per fact: P(correct) on a probe question
              scored against alternatives (hard recall gate, deterministic)
  2. DECIDE   consolidation gate state machine per fact:
                already_known -> never written
                learning (<90%) -> full write, weight importance*(1+emotion)
                consolidated (>=90%) -> rehearsal write shrinking toward 100%
                mastered -> skipped; relapse -> back to learning
              every decision -> receipts.jsonl
  3. WRITE    train memory only, per-sample weighted by the decision, plus:
              - recall gate hinges (familiarity high on facts, low on text)
              - do-no-harm KL on general text vs the memory-free base
              - slot balance penalty (anti-hub)

    python scripts/phase2_gated.py --save
    python scripts/phase2_gated.py --no-recall-gate --no-write-gate   # ablation
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dmodel.config import ExperimentConfig  # noqa: E402
from dmodel.data import GENERAL_QA, GENERAL_TEXT, make_facts, recall_questions  # noqa: E402
from dmodel.data.episodes import make_episodes  # noqa: E402
from dmodel.data.negatives import load_negatives  # noqa: E402
from dmodel.eval import confidences, mc_accuracy, mean_nll  # noqa: E402
from dmodel.gates import ConsolidationGate, ReceiptLog  # noqa: E402
from dmodel.gates import aggregate_cues as aggregate  # noqa: E402
from dmodel.gates.write_gate import _value_positions  # noqa: E402
from dmodel.memory import (  # noqa: E402
    attach_memory, set_memory_enabled, set_recall_mode, slot_balance_loss, slot_usage,
)
from dmodel.model import load_model  # noqa: E402


# --------------------------------------------------------------------------
# eval helpers
# --------------------------------------------------------------------------
@torch.no_grad()
def gate_open_rate(model, tok, blocks, prompts: list[str], device, last_only: bool) -> dict:
    """Fraction of tokens where the hard recall gate is open, per layer."""
    rates = {k: [] for k in blocks}
    fams = {k: [] for k in blocks}
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        model(input_ids=ids)
        for k, b in blocks.items():
            if b.last_recall is None:
                continue
            g, f = b.last_recall[0], b.last_readout.familiarity[0]
            if last_only:
                g, f = g[-1:], f[-1:]
            rates[k].append(g.float().mean().item())
            fams[k].append(f.float().mean().item())
    return {k: {"open_rate": sum(v) / max(len(v), 1),
                "mean_familiarity": sum(fams[k]) / max(len(fams[k]), 1)}
            for k, v in rates.items()}


def evaluate(model, tok, blocks, questions, kind_of, device, label, rg_on) -> dict:
    t0 = time.time()
    model.eval()
    set_recall_mode(blocks, "hard" if rg_on else "off")
    res = {}
    for kind in ("important", "trivial"):
        for qk in ("seen", "unseen", "unseen_rf"):
            qs = [q for q in questions if q.kind == qk and kind_of[q.fact.entity] == kind]
            res[f"{kind}_{qk}"] = mc_accuracy(model, tok, qs, device)["accuracy"]
    unseen_imp = [q for q in questions
                  if q.kind == "unseen" and kind_of[q.fact.entity] == "important"]
    conf_u = confidences(model, tok, unseen_imp, device)
    res["important_unseen_mean_conf"] = sum(conf_u) / max(len(conf_u), 1)
    res["general_qa"] = mc_accuracy(model, tok, GENERAL_QA, device)["accuracy"]
    res["general_nll"] = mean_nll(model, tok, GENERAL_TEXT, device)
    imp_q = [q.prompt for q in questions if kind_of[q.fact.entity] == "important"][:200]
    res["gate_fact_questions"] = gate_open_rate(model, tok, blocks, imp_q, device, last_only=True)
    res["gate_general_text"] = gate_open_rate(model, tok, blocks, GENERAL_TEXT, device, last_only=False)

    def rate(d):
        return " ".join(f"L{k}:{v['open_rate']:.0%}" for k, v in d.items())

    print(f"[{label}] important seen {res['important_seen']:.1%} unseen {res['important_unseen']:.1%} "
          f"unseen-relation-first {res['important_unseen_rf']:.1%} "
          f"(unseen conf {res['important_unseen_mean_conf']:.1%}) | trivial seen "
          f"{res['trivial_seen']:.1%} unseen {res['trivial_unseen']:.1%} | "
          f"general QA {res['general_qa']:.1%} NLL {res['general_nll']:.3f}")
    print(f"[{label}] recall gate open - fact questions: {rate(res['gate_fact_questions'])} | "
          f"general text: {rate(res['gate_general_text'])}  ({time.time() - t0:.0f}s)")
    return res


# --------------------------------------------------------------------------
# training helpers
# --------------------------------------------------------------------------
def encode(tok, texts, max_len=None):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
              return_offsets_mapping=True, truncation=max_len is not None,
              max_length=max_len)
    offsets = enc.pop("offset_mapping").tolist()
    return enc, offsets


def top_slots(block, j: int, positions: list[int], n: int = 4) -> list[int]:
    ro = block.last_readout
    idx = ro.indices[j, positions].flatten()
    w = ro.weights[j, positions].flatten()
    out, seen = [], set()
    for o in w.argsort(descending=True).tolist():
        s = idx[o].item()
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == n:
            break
    return out


def weighted_lm_loss(logits, ids, mask, weights):
    """Per-sequence mean token CE, weighted per sample (write strength)."""
    b, t, v = logits.shape
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, v), labels.reshape(-1),
                         ignore_index=-100, reduction="none").view(b, t - 1)
    valid = (labels != -100).float()
    per_seq = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    return (weights * per_seq).sum() / b, per_seq.mean().detach()


LEARN_REASONS = ("learning", "relearning", "gate_disabled")


def lr_schedule(lr: float, epoch: int, epochs: int, min_frac: float) -> float:
    """Cosine decay from lr to lr*min_frac: slots grow less plastic over time."""
    t = (epoch - 1) / max(epochs - 1, 1)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def make_write_batches(writes, bs: int, rng: random.Random) -> list:
    """Learning and rehearsal writes go in separate batches, so each batch has
    one write strength and its step size can be scaled cleanly. Batch order
    is shuffled so the two kinds interleave."""
    learn = [x for x in writes if x[1].reason in LEARN_REASONS]
    reh = [x for x in writes if x[1].reason not in LEARN_REASONS]
    rng.shuffle(learn)
    rng.shuffle(reh)
    batches = [grp[i : i + bs] for grp in (learn, reh) for i in range(0, len(grp), bs)]
    rng.shuffle(batches)
    return batches


def hub_stats(usage: torch.Tensor) -> dict:
    total = usage.sum().clamp_min(1e-9)
    p = usage / total
    nz = p[p > 0]
    ent = -(nz * nz.log()).sum().item()
    return {"top10_share": usage.topk(10).values.sum().item() / total.item(),
            "slots_used": int((usage > 0).sum().item()),
            "norm_entropy": ent / torch.log(torch.tensor(float(max(len(nz), 2)))).item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--entities", type=int, default=100)
    ap.add_argument("--trivial-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--neg-batch", type=int, default=8)
    ap.add_argument("--neg-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-recall-gate", action="store_true")
    ap.add_argument("--no-write-gate", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    rg, wg = cfg.recall_gate, cfg.write_gate
    rg_on = rg.enabled and not args.no_recall_gate
    wg_on = wg.enabled and not args.no_write_gate
    device = torch.device(args.device or cfg.train_device)
    lr = args.lr or cfg.train.lr
    bs = args.batch or cfg.train.batch_size
    epochs = args.epochs or cfg.train.epochs
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    facts = make_facts(args.entities, seed=args.seed)
    episodes, kind_of, probes = make_episodes(facts, args.trivial_frac, seed=args.seed)
    fact_keys = sorted(probes)
    kind_by_key = {e.fact_key: e.kind for e in episodes}
    questions = recall_questions(facts)
    negatives, neg_src = load_negatives(seed=args.seed)
    print(f"experiences: {dict(Counter(e.kind for e in episodes))} | facts: "
          f"{dict(Counter(kind_by_key.values()))} | negatives: {len(negatives)} ({neg_src})")
    print(f"recall gate: {'ON thr=' + str(rg.threshold) if rg_on else 'OFF'} | "
          f"consolidation gate: {'ON conf>=' + str(wg.confidence_threshold) if wg_on else 'OFF'} | "
          f"balance weight {cfg.train.balance_weight}")

    model, tok = load_model(cfg, device)
    blocks = attach_memory(model, cfg.memory, recall_threshold=rg.threshold,
                           recall_mode="hard" if rg_on else "off")
    n_slots = cfg.memory.n_slots
    print(f"memory: layers {cfg.memory.insert_layers}, "
          f"{sum(p.numel() for p in blocks.parameters()) / 1e6:.0f}M params")

    before = evaluate(model, tok, blocks, questions, kind_of, device, "before", rg_on)

    gate_params = [b.gate for b in blocks.values()]
    gid = {id(p) for p in gate_params}
    opt = torch.optim.AdamW(
        [{"params": [p for p in blocks.parameters() if id(p) not in gid], "lr": lr},
         {"params": gate_params, "lr": lr * 10}], weight_decay=0.0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"phase2_{stamp}"
    receipts = ReceiptLog(run_dir / "receipts.jsonl")
    gate = ConsolidationGate(wg.confidence_threshold, wg.relapse_threshold,
                             wg.rehearsal_scale, wg.min_weight, wg.importance_floor,
                             enabled=wg_on, confirm_epochs=wg.confirm_epochs)
    print(f"write strength -> step size: {cfg.train.write_step_scaling} | LR cosine decay to "
          f"{cfg.train.lr_min_frac:.0%} | confirm_epochs {wg.confirm_epochs}")
    neg_iter = itertools.cycle(negatives)
    epoch_log, step = [], 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # ---- 1. probe confidence per fact -------------------------------
        model.eval()
        set_recall_mode(blocks, "hard" if rg_on else "off")
        flat = [(k, q) for k in fact_keys for q in probes[k]]
        cue_conf = confidences(model, tok, [q for _, q in flat], device)
        per_key: dict[str, list[float]] = {}
        for (k, _), c in zip(flat, cue_conf):
            per_key.setdefault(k, []).append(c)
        confs = {k: aggregate(v, wg.cue_aggregate) for k, v in per_key.items()}
        mean_cue = {k: sum(v) / len(v) for k, v in per_key.items()}
        min_cue = {k: min(v) for k, v in per_key.items()}
        imp_multi = [k for k in fact_keys
                     if kind_by_key[k] == "important" and len(per_key[k]) > 1]
        by_cue = [sum(per_key[k][i] for k in imp_multi) / max(len(imp_multi), 1)
                  for i in range(max((len(per_key[k]) for k in imp_multi), default=0))]
        for k in fact_keys:
            gate.update_fact(k, confs[k], epoch)

        # ---- 2. decide -------------------------------------------------
        decisions = [gate.decide(e, confs[e.fact_key]) for e in episodes]
        writes = [(e, d) for e, d in zip(episodes, decisions) if d.accepted]
        for d in decisions:
            if not d.accepted:
                receipts.write(d, epoch)
        reasons = Counter(d.reason for d in decisions)
        imp_conf = [confs[k] for k in fact_keys if kind_by_key[k] == "important"]
        conf_mean = sum(imp_conf) / len(imp_conf)
        cue_mean = sum(mean_cue[k] for k in fact_keys
                       if kind_by_key[k] == "important") / len(imp_conf)
        cue_min = sum(min_cue[k] for k in fact_keys
                      if kind_by_key[k] == "important") / len(imp_conf)
        conf_90 = sum(c >= wg.confidence_threshold for c in imp_conf) / len(imp_conf)

        if not writes:
            print(f"  epoch {epoch:>3}: nothing to write -> stopping. {dict(reasons)}")
            epoch_log.append({"epoch": epoch, "writes": 0, "reasons": dict(reasons),
                              "states": gate.summary(kind_by_key)})
            break

        # ---- 3. write ----------------------------------------------------
        if rg_on:
            frac = (epoch - 1) / max(epochs - 1, 1)
            set_recall_mode(blocks, "soft", rg.temp_start * (rg.temp_end / rg.temp_start) ** frac)
        model.train()
        lr_epoch = lr_schedule(lr, epoch, epochs, cfg.train.lr_min_frac)
        batches = make_write_batches(writes, bs, rng)
        sums, n_steps = Counter(), 0
        usage = {k: torch.zeros(n_slots, device=device) for k in blocks}
        for batch in batches:
            enc, offs = encode(tok, [e.text for e, _ in batch])
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            w = torch.tensor([d.weight for _, d in batch], device=device)
            # write strength -> optimizer step size. Adam normalises gradient
            # magnitude, so scaling only the loss would NOT make a rehearsal
            # light; scaling the learning rate does.
            if cfg.train.write_step_scaling:
                step_scale = min(1.0, w.mean().item())
                w = w / w.mean()             # keep relative weights within the batch
            else:
                step_scale = 1.0
            opt.param_groups[0]["lr"] = lr_epoch * step_scale
            opt.param_groups[1]["lr"] = lr_epoch * 10 * step_scale
            sums["step_scale"] += step_scale
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss, raw_lm = weighted_lm_loss(logits, ids, mask, w)
            del logits
            sums["lm"] += raw_lm.item()

            vpos = [[p - 1 for p in _value_positions(e.text, e.value, offs[j]) if p > 0]
                    for j, (e, _) in enumerate(batch)]
            for j, (e, d) in enumerate(batch):
                slots = {k: top_slots(b, j, vpos[j]) for k, b in blocks.items()} if vpos[j] else None
                receipts.write(d, epoch, slots=slots, step=step)

            bmask = mask.bool()
            if cfg.train.balance_weight > 0:
                bal = torch.stack([slot_balance_loss(b.last_readout, bmask, n_slots)
                                   for b in blocks.values()]).mean()
                loss = loss + cfg.train.balance_weight * bal
                sums["balance"] += bal.item()
            with torch.no_grad():
                for k, b in blocks.items():
                    usage[k] += slot_usage(b.last_readout, bmask, n_slots)

            if rg_on and any(vpos):
                pos_h = torch.stack([
                    F.relu(rg.threshold + rg.margin - torch.cat(
                        [b.last_readout.familiarity[j, torch.tensor(v, device=device)]
                         for j, v in enumerate(vpos) if v])).mean()
                    for b in blocks.values()]).mean()

                nenc, _ = encode(tok, [next(neg_iter) for _ in range(args.neg_batch)],
                                 max_len=args.neg_len)
                nids = nenc["input_ids"].to(device)
                nmask = nenc["attention_mask"].to(device)
                with torch.no_grad():
                    set_memory_enabled(blocks, False)
                    base_logp = F.log_softmax(
                        model(input_ids=nids, attention_mask=nmask).logits.float(), -1)
                    set_memory_enabled(blocks, True)
                mem_logp = F.log_softmax(
                    model(input_ids=nids, attention_mask=nmask).logits.float(), -1)
                valid = nmask.bool()
                kl = (base_logp.exp() * (base_logp - mem_logp)).sum(-1)[valid].mean()
                neg_h = torch.stack([
                    F.relu(b.last_readout.familiarity[valid] - (rg.threshold - rg.margin)).mean()
                    for b in blocks.values()]).mean()
                loss = loss + rg.hinge_weight * (pos_h + neg_h) + rg.kl_weight * kl
                sums["pos_h"] += pos_h.item()
                sums["neg_h"] += neg_h.item()
                sums["kl"] += kl.item()
                del base_logp, mem_logp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            n_steps += 1

        avg = {k: round(v / n_steps, 4) for k, v in sums.items()}
        hubs = {k: hub_stats(u) for k, u in usage.items()}
        states = gate.summary(kind_by_key)
        imp_states = {k.split(":")[1]: v for k, v in states.items() if k.startswith("important:")}
        n_learn = sum(1 for _, d in writes if d.reason in ("learning", "relearning"))
        n_reh = sum(1 for _, d in writes if d.reason == "rehearsal")
        n_slip = sum(1 for _, d in writes if d.reason == "rehearsal_slipping")
        epoch_log.append({"epoch": epoch, "writes": len(writes), "learn_writes": n_learn,
                          "rehearsals": n_reh, "slipping_rehearsals": n_slip,
                          "reasons": dict(reasons), "states": states,
                          "important_conf_mean": conf_mean, "important_cue_mean": cue_mean,
                          "important_cue_min": cue_min, "important_conf_by_cue": by_cue,
                          "important_conf_ge_thr": conf_90,
                          "losses": avg, "hubs": hubs,
                          "lr": lr_epoch,
                          "scale": {k: round(b.gate.item(), 3) for k, b in blocks.items()}})
        cue_txt = "/".join(f"{c:.0%}" for c in by_cue)
        print(f"  epoch {epoch:>3}: conf {wg.cue_aggregate} {conf_mean:.1%} "
              f"[min {cue_min:.0%} avg {cue_mean:.0%}] cues(trained/p1/p2) {cue_txt} "
              f"(>=90%: {conf_90:.0%}) | learn {n_learn} rehearse {n_reh} slip {n_slip} | "
              f"important {imp_states} | " +
              " ".join(f"{k} {v:.3f}" for k, v in avg.items()) + " | hub top10 " +
              " ".join(f"L{k}:{h['top10_share']:.1%}" for k, h in hubs.items()) +
              f" | lr {lr_epoch:.1e} ({time.time() - t0:.0f}s)")

    receipts.close()
    after = evaluate(model, tok, blocks, questions, kind_of, device, "after", rg_on)

    # calibration: does the system's own sense of knowing match reality?
    imp_keys = [k for k in fact_keys if kind_by_key[k] == "important"]
    flat = [(k, q) for k in imp_keys for q in probes[k]]
    fc: dict[str, list[float]] = {}
    for (k, _), c in zip(flat, confidences(model, tok, [q for _, q in flat], device)):
        fc.setdefault(k, []).append(c)
    self_min = sum(min(v) for v in fc.values()) / len(fc)
    self_avg = sum(sum(v) / len(v) for v in fc.values()) / len(fc)
    self_gate = sum(aggregate(v, wg.cue_aggregate) for v in fc.values()) / len(fc)
    after["self_assessed_min_cue"] = self_min
    after["self_assessed_avg_cue"] = self_avg
    after["self_assessed_gate"] = self_gate

    final_states = gate.summary(kind_by_key)
    relapses = sum(st.relapses for st in gate.facts.values())
    displaced = sum(st.displaced for st in gate.facts.values())
    print("\n=== PHASE 2 RESULT ===")
    for key in ("important_seen", "important_unseen", "important_unseen_rf",
                "trivial_seen", "trivial_unseen", "general_qa"):
        print(f"{key:<19}: {before[key]:.1%} -> {after[key]:.1%}")
    print(f"{'general_nll':<17}: {before['general_nll']:.3f} -> {after['general_nll']:.3f} "
          f"({after['general_nll'] - before['general_nll']:+.3f})")
    print(f"unseen confidence: {before['important_unseen_mean_conf']:.1%} -> "
          f"{after['important_unseen_mean_conf']:.1%}")
    total_writes = sum(e.get("writes", 0) for e in epoch_log)
    print(f"fact states      : {final_states}")
    print(f"relapses         : {relapses} (consolidated -> <{wg.relapse_threshold:.0%}) | "
          f"displaced: {displaced} (already_known -> <{wg.relapse_threshold:.0%})")
    print(f"calibration      : self-assessed {self_gate:.1%} ({wg.cue_aggregate}) "
          f"[min {self_min:.1%} / avg {self_avg:.1%}] vs actual unseen accuracy "
          f"{after['important_unseen']:.1%} / relation-first {after['important_unseen_rf']:.1%} "
          f"(unseen conf {after['important_unseen_mean_conf']:.1%})")
    gt = after["gate_general_text"]
    print("general-text gate: " + " ".join(
        f"L{k}:{before['gate_general_text'][k]['open_rate']:.0%}->{v['open_rate']:.0%}"
        for k, v in gt.items()) + "  (watch: relation-first training must not open it)")
    print(f"total writes     : {total_writes} over {len(epoch_log)} epochs "
          f"(ungated would be {len(episodes) * len(epoch_log)})")

    report = {"args": vars(args), "base_model": cfg.base_model, "memory": vars(cfg.memory),
              "recall_gate": vars(rg) | {"active": rg_on},
              "write_gate": vars(wg) | {"active": wg_on},
              "train": vars(cfg.train), "negatives": neg_src,
              "before": before, "after": after, "epochs": epoch_log,
              "final_states": final_states, "relapses": relapses, "displaced": displaced,
              "facts": {k: vars(st) for k, st in gate.facts.items()},
              "total_writes": total_writes, "train_seconds": time.time() - t_start}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report  : {run_dir / 'report.json'}")
    print(f"receipts: {run_dir / 'receipts.jsonl'}")

    if args.save:
        ckpt = Path("checkpoints")
        ckpt.mkdir(exist_ok=True)
        path = ckpt / f"phase2_memory_{stamp}.pt"
        torch.save({"memory_config": vars(cfg.memory), "recall_gate": vars(rg),
                    "blocks": blocks.state_dict(), "fact_states": {k: vars(s) for k, s in gate.facts.items()},
                    "seed": args.seed, "entities": args.entities}, path)
        print(f"memory saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
