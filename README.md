# Deterministic Model

A hybrid mind model: stochastic neural processing governed by deterministic logic gates.

**Core principle:** the stochastic side is free to be creative; the deterministic side owns what gets asserted, remembered, and trusted.

## Architecture (human analogue)

| Human                       | System                                            |
|-----------------------------|---------------------------------------------------|
| Working memory              | Context window                                    |
| Hippocampus (fast learning) | Sparse parametric memory module (plastic)         |
| Neocortex (slow, stable)    | Frozen base model weights                         |
| Sleep consolidation         | Offline write / replay / compression cycles       |
| Executive inhibition        | Deterministic gates (write, recall, abstention)   |

## Roadmap

1. **Foundation** — env check, frozen base + sparse memory adapter, cross-session recall test, base-skill regression benchmarks.
2. **Experience** — episodic encoding, salience-gated writes (surprise × emotion × importance), consolidation with replay.
3. **Trust** — epistemic ledger (provenance + write receipts), abstention gate.
4. **Scale** — larger base (7–8B, 4-bit), long-lived memory tests.

## Hardware

- RTX 3090 Ti (24GB) — primary training
- RTX 5060 Ti (16GB, Blackwell) — eval / baselines / embeddings
- 128GB system RAM — episode buffers, replay, optimizer offload

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
# Blackwell (5060 Ti) needs a CUDA 12.8+ build of PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/env_check.py
python -m pytest tests -q
```

## Layout

```
dmodel/
  memory/sparse_memory.py   # product-key sparse memory (learned LTM)
  config.py                 # dataclass configs
configs/phase1.yaml         # Phase 1 experiment config
scripts/env_check.py        # GPU / CUDA / bf16 verification
tests/                      # CPU-runnable unit tests
```
