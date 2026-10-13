"""Phase 1: can the sparse memory act as long-term memory?

1. TEACH   - train ONLY the memory blocks on invented facts (base frozen)
2. CLEAR   - every eval is a fresh forward pass; facts are never in context
3. RECALL  - multiple-choice recall on trained ("seen") and never-trained
             ("unseen") question phrasings, before vs after training
4. REGRESS - general-knowledge QA accuracy + plain-text NLL, before vs after

"Before" is measured with memory attached but gates at zero, which is
mathematically identical to the untouched base model.

    python scripts/phase1_recall.py                  # defaults
    python scripts/phase1_recall.py --entities 50 --epochs 10   # quick run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dmodel.config import ExperimentConfig  # noqa: E402
from dmodel.data import (  # noqa: E402
    GENERAL_QA, GENERAL_TEXT, make_facts, recall_questions, training_texts,
)
from dmodel.eval import mc_accuracy, mean_nll  # noqa: E402
from dmodel.memory import attach_memory  # noqa: E402


def load_model(cfg: ExperimentConfig, device):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # causal: real tokens never see padding
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.base_model, dtype=getattr(torch, cfg.dtype), device_map={"": device}
    )
    return model, tok


def evaluate(model, tok, questions, device, label: str) -> dict:
    t0 = time.time()
    model.eval()
    seen = [q for q in questions if q.kind == "seen"]
    unseen = [q for q in questions if q.kind == "unseen"]
    res = {
        "recall_seen": mc_accuracy(model, tok, seen, device),
        "recall_unseen": mc_accuracy(model, tok, unseen, device),
        "general_qa": mc_accuracy(model, tok, GENERAL_QA, device),
        "general_nll": mean_nll(model, tok, GENERAL_TEXT, device),
    }
    print(f"[{label}] recall seen {res['recall_seen']['accuracy']:.1%} | "
          f"unseen {res['recall_unseen']['accuracy']:.1%} | "
          f"general QA {res['general_qa']['accuracy']:.1%} | "
          f"general NLL {res['general_nll']:.3f}  ({time.time() - t0:.0f}s)")
    return res


def batches(texts: list[str], tok, bs: int, rng: random.Random):
    order = texts[:]
    rng.shuffle(order)
    for i in range(0, len(order), bs):
        enc = tok(order[i : i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        yield enc["input_ids"], enc["attention_mask"], labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--entities", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="save memory weights")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(args.device or cfg.train_device)
    lr = args.lr or cfg.train.lr
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    facts = make_facts(args.entities, seed=args.seed)
    texts = training_texts(facts)
    questions = recall_questions(facts)
    n_choices = len(questions[0].choices)
    print(f"facts: {len(facts)} ({args.entities} entities) | train sentences: "
          f"{len(texts)} | chance level: {1 / n_choices:.1%}")
    print(f"example: {texts[0]!r}")

    model, tok = load_model(cfg, device)
    blocks = attach_memory(model, cfg.memory)
    n_mem = sum(p.numel() for p in blocks.parameters())
    print(f"memory: layers {cfg.memory.insert_layers}, {n_mem / 1e6:.0f}M trainable params")

    before = evaluate(model, tok, questions, device, "before")

    # ---- TEACH: train memory only ------------------------------------
    # gates start at 0 (exact identity); give them a 10x lr so the memory
    # path opens quickly instead of crawling out of zero
    gate_params = [b.gate for b in blocks.values()]
    gate_ids = {id(p) for p in gate_params}
    other = [p for p in blocks.parameters() if id(p) not in gate_ids]
    opt = torch.optim.AdamW(
        [{"params": other, "lr": lr}, {"params": gate_params, "lr": lr * 10}],
        weight_decay=0.0,
    )
    usage = {k: torch.zeros(cfg.memory.n_slots, device=device) for k in blocks}
    steps_per_epoch = -(-len(texts) // args.batch)
    print(f"training: {args.epochs} epochs x {steps_per_epoch} steps, lr {lr}")
    t0 = time.time()
    log = []
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for ids, mask, labels in batches(texts, tok, args.batch, rng):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            total += out.loss.item()
            n += 1
            with torch.no_grad():
                for k, b in blocks.items():
                    idx = b.last_readout.indices[mask.bool()].flatten()
                    usage[k] += torch.bincount(idx, minlength=cfg.memory.n_slots).float()
        gates = {k: round(b.gate.item(), 4) for k, b in blocks.items()}
        log.append({"epoch": epoch + 1, "loss": total / n, "gates": gates})
        print(f"  epoch {epoch + 1:>3}/{args.epochs}  loss {total / n:.4f}  "
              f"gates {gates}  ({time.time() - t0:.0f}s)")

    after = evaluate(model, tok, questions, device, "after")

    # ---- slot usage: how much of the memory was actually touched -------
    slot_stats = {}
    for k, u in usage.items():
        used = (u > 0).sum().item()
        top = u.sort(descending=True).values
        top1pct = top[: max(1, len(top) // 100)].sum().item() / max(u.sum().item(), 1)
        slot_stats[k] = {"slots_used": used,
                         "fraction_used": used / cfg.memory.n_slots,
                         "top1pct_share": top1pct}
        print(f"slots layer {k}: {used:,}/{cfg.memory.n_slots:,} used "
              f"({used / cfg.memory.n_slots:.1%}), top 1% of slots carry {top1pct:.1%} of reads")

    # ---- report ---------------------------------------------------------
    def strip(r):  # drop per-question lists from the summary
        return {k: ({kk: vv for kk, vv in v.items() if kk != "per_question"}
                    if isinstance(v, dict) else v) for k, v in r.items()}

    d_gen = after["general_qa"]["accuracy"] - before["general_qa"]["accuracy"]
    d_nll = after["general_nll"] - before["general_nll"]
    print("\n=== PHASE 1 RESULT ===")
    print(f"recall (seen)   : {before['recall_seen']['accuracy']:.1%} -> {after['recall_seen']['accuracy']:.1%}")
    print(f"recall (unseen) : {before['recall_unseen']['accuracy']:.1%} -> {after['recall_unseen']['accuracy']:.1%}")
    print(f"general QA      : {before['general_qa']['accuracy']:.1%} -> {after['general_qa']['accuracy']:.1%} ({d_gen:+.1%})")
    print(f"general NLL     : {before['general_nll']:.3f} -> {after['general_nll']:.3f} ({d_nll:+.3f})")

    run_dir = Path("runs")
    run_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "args": vars(args), "base_model": cfg.base_model,
        "memory": vars(cfg.memory), "chance": 1 / n_choices,
        "before": strip(before), "after": strip(after),
        "train_log": log, "slot_usage": slot_stats,
        "train_seconds": time.time() - t0,
    }
    out_path = run_dir / f"phase1_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"report: {out_path}")

    if args.save:
        ckpt = Path("checkpoints")
        ckpt.mkdir(exist_ok=True)
        path = ckpt / f"phase1_memory_{stamp}.pt"
        torch.save({"memory_config": vars(cfg.memory), "blocks": blocks.state_dict(),
                    "facts_seed": args.seed, "entities": args.entities}, path)
        print(f"memory saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
