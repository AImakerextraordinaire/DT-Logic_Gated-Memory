"""Attach sparse memory blocks to a frozen Hugging Face decoder model.

The base model's weights are frozen (the "neocortex"); only the memory blocks
(the "hippocampus") are trainable. Blocks are inserted via forward hooks on
the chosen decoder layers, so no model code is modified and detaching
restores the original model exactly.
"""
from __future__ import annotations

import torch.nn as nn

from ..config import MemoryConfig
from .sparse_memory import MemoryBlock


_LAYER_PATHS = (
    # multimodal wrappers (Qwen3.5, Muse Glimmer, ...) first: their text
    # decoder lives one level deeper than plain causal LMs
    "model.language_model.layers",
    "language_model.model.layers",
    "model.text_model.layers",
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "layers",
)


def _hidden_size(model: nn.Module) -> int:
    cfg = model.config
    if hasattr(cfg, "get_text_config"):
        cfg = cfg.get_text_config()
    elif hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    return cfg.hidden_size


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError("Could not locate decoder layers on this model")


def _make_hook(block: MemoryBlock):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (block(output[0]),) + tuple(output[1:])
        return block(output)

    return hook


def attach_memory(
    model: nn.Module,
    cfg: MemoryConfig,
    freeze_base: bool = True,
    recall_threshold: float = 0.8,
    recall_mode: str = "off",
) -> nn.ModuleDict:
    """Insert MemoryBlocks after `cfg.insert_layers`. Returns the blocks."""
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)

    layers = _decoder_layers(model)
    d_model = _hidden_size(model)
    blocks = nn.ModuleDict()
    handles = []

    for idx in cfg.insert_layers:
        if not 0 <= idx < len(layers):
            raise IndexError(f"insert layer {idx} out of range (0..{len(layers) - 1})")
        ref = next(layers[idx].parameters())
        block = MemoryBlock(
            d_model,
            n_keys=cfg.n_keys,
            key_dim=cfg.key_dim,
            topk=cfg.topk,
            heads=cfg.heads,
            value_dim=cfg.value_dim,
            normalize_keys=cfg.normalize_keys,
            recall_threshold=recall_threshold,
            recall_mode=recall_mode,
        ).to(device=ref.device)  # memory kept in fp32 for stable learning
        blocks[str(idx)] = block
        handles.append(layers[idx].register_forward_hook(_make_hook(block)))

    model._memory_blocks = blocks
    model._memory_handles = handles
    return blocks


def detach_memory(model: nn.Module) -> None:
    """Remove hooks; the base model returns to its original behaviour."""
    for h in getattr(model, "_memory_handles", []):
        h.remove()
    model._memory_handles = []
    model._memory_blocks = None
