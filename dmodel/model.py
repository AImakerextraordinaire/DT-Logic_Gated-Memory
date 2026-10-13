from __future__ import annotations

import torch

from .config import ExperimentConfig


def load_model(cfg: ExperimentConfig, device):
    """Load the frozen base (multimodal wrapper) + tokenizer onto one device."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # causal: real tokens never see padding
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.base_model, dtype=getattr(torch, cfg.dtype), device_map={"": device}
    )
    return model, tok
