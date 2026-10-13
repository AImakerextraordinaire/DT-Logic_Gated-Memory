"""Per-cue diagnostic for a saved Phase 2 memory checkpoint.

For each attribute and each cue (trained QA, probe rewordings, eval unseen)
reports, over the important facts:
  acc / conf      accuracy and mean P(correct) with memory (hard recall gate)
  base conf       same with memory disabled (what the base model guesses)
  gate L15 / L23  how often the recall gate is open at the cue's last token
  fam L15 / L23   mean familiarity there (gate threshold from config)

    python scripts/diagnose_cues.py checkpoints/phase2_memory_XXXX.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dmodel.config import ExperimentConfig  # noqa: E402
from dmodel.data.episodes import make_episodes  # noqa: E402
from dmodel.data.facts import ATTRIBUTES, _question, make_facts  # noqa: E402
from dmodel.eval import batch_choice_logprobs, confidences  # noqa: E402
from dmodel.memory import attach_memory, set_memory_enabled, set_recall_mode  # noqa: E402
from dmodel.model import load_model  # noqa: E402


@torch.no_grad()
def gate_at_last_token(model, tok, blocks, prompts, device):
    open_, fam = {k: 0.0 for k in blocks}, {k: 0.0 for k in blocks}
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        model(input_ids=ids)
        for k, b in blocks.items():
            open_[k] += b.last_recall[0, -1].item()
            fam[k] += b.last_readout.familiarity[0, -1].item()
    n = max(len(prompts), 1)
    return {k: v / n for k, v in open_.items()}, {k: v / n for k, v in fam.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device or cfg.train_device)
    seed, entities = ckpt.get("seed", 0), ckpt.get("entities", 100)

    facts = make_facts(entities, seed=seed)
    _, kind_of, _ = make_episodes(facts, 0.3, seed=seed)
    important = [f for f in facts if kind_of[f.entity] == "important"]

    model, tok = load_model(cfg, device)
    blocks = attach_memory(model, cfg.memory, recall_threshold=cfg.recall_gate.threshold)
    blocks.load_state_dict(ckpt["blocks"])
    model.eval()
    set_recall_mode(blocks, "hard")
    thr = cfg.recall_gate.threshold

    print(f"checkpoint: {args.checkpoint} | important facts: {len(important)} | gate thr {thr}")
    lk = list(blocks)  # memory layer keys, e.g. ["15", "23"]
    print(f"{'attribute':<11} {'cue':<52} {'acc':>6} {'conf':>6} {'base':>6} " +
          " ".join(f"{'gate' + k:>7}" for k in lk) + " " +
          " ".join(f"{'fam' + k:>6}" for k in lk))
    for attr, spec in ATTRIBUTES.items():
        fs = [f for f in important if f.attribute == attr]
        seen_qa = next(t for t in spec["train"] if t.startswith("Q:"))
        cues = [("trained", seen_qa)] + [(f"probe{i + 1}", t) for i, t in enumerate(spec["probes"])] \
            + [("UNSEEN eval", spec["unseen"]), ("UNSEEN rel-first", spec["unseen_rf"])]
        for label, tpl in cues:
            qs = [_question(f, tpl, label) for f in fs]
            conf = confidences(model, tok, qs, device)
            sc =batch_choice_logprobs(model, tok, qs, device)
            acc = sum(max(range(len(s)), key=s.__getitem__) == q.answer_idx
                      for q, s in zip(qs, sc)) / len(qs)
            g, fam = gate_at_last_token(model, tok, blocks, [q.prompt for q in qs], device)
            set_memory_enabled(blocks, False)
            base = confidences(model, tok, qs, device)
            set_memory_enabled(blocks, True)
            shown = tpl.split("{v}")[0].replace("\n", "\\n").strip()
            print(f"{attr:<11} {label + ': ' + shown:<52.52} {acc:>6.1%} "
                  f"{sum(conf) / len(conf):>6.1%} {sum(base) / len(base):>6.1%} " +
                  " ".join(f"{g[k]:>7.0%}" for k in lk) + " " +
                  " ".join(f"{fam[k]:>6.2f}" for k in lk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
