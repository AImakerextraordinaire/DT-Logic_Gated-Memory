"""Load the real base model, attach memory, and verify on GPU.

Checks: transformers can load the model, memory attaches to the text decoder,
output is unchanged at init (identity), a backward pass reaches the memory,
and reports VRAM use. Run from the project root:

    python scripts/load_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

from dmodel.config import ExperimentConfig  # noqa: E402
from dmodel.memory import attach_memory  # noqa: E402


def gb(x: int) -> str:
    return f"{x / 1024**3:.2f} GB"


def main() -> int:
    cfg = ExperimentConfig.from_yaml("configs/phase1.yaml")
    dev = torch.device(cfg.train_device)
    print(f"transformers : {transformers.__version__}")
    print(f"base model   : {cfg.base_model}")
    print(f"device       : {dev} ({torch.cuda.get_device_name(dev)})")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.base_model, dtype=getattr(torch, cfg.dtype), device_map={"": dev}
    ).eval()
    print(f"loaded in    : {time.time() - t0:.1f}s, VRAM {gb(torch.cuda.memory_allocated(dev))}")

    ids = tok("The cathedral isn't just code, it's", return_tensors="pt").to(dev)
    with torch.no_grad():
        before = model(**ids).logits.float()

    blocks = attach_memory(model, cfg.memory)
    n_mem = sum(p.numel() for p in blocks.parameters())
    print(f"memory       : layers {cfg.memory.insert_layers}, "
          f"{cfg.memory.n_slots:,} slots/layer, {n_mem / 1e6:.0f}M params")

    with torch.no_grad():
        after = model(**ids).logits.float()
    diff = (after - before).abs().max().item()
    print(f"identity     : max |delta logits| = {diff:.2e} -> {'OK' if diff < 1e-3 else 'FAIL'}")

    # open the gates slightly so gradients reach every memory parameter
    with torch.no_grad():
        for b in blocks.values():
            b.gate.fill_(1e-2)
    torch.cuda.reset_peak_memory_stats(dev)
    out = model(**ids, labels=ids["input_ids"])
    out.loss.backward()
    reached = all(b.memory.query.weight.grad is not None for b in blocks.values())
    print(f"backward     : loss {out.loss.item():.3f}, grads reach memory -> "
          f"{'OK' if reached else 'FAIL'}")
    print(f"peak VRAM    : {gb(torch.cuda.max_memory_allocated(dev))} "
          f"of {gb(torch.cuda.get_device_properties(dev).total_memory)}")

    ok = diff < 1e-3 and reached
    print("RESULT:", "ALL GOOD" if ok else "ISSUES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
