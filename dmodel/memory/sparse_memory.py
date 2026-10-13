"""Sparse product-key memory: the learned, parametric long-term memory.

Based on product-key memory (Lample et al., 2019) and memory layers at scale
(Meta, 2024). A huge table of value slots, of which only `topk` per head are
touched per token, so capacity grows without growing per-step compute.

Product keys: the query is split in half; each half is scored against
`n_keys` sub-keys. The Cartesian product of the two top-k lists addresses
n_keys**2 slots while only scoring 2 * n_keys keys.

Phase 2 additions
-----------------
normalize_keys: query halves and sub-keys are L2-normalised, so each slot's
    score is a sum of two cosines in [-2, 2]. That bounded score doubles as a
    *familiarity* signal ("have I stored something about this?") whose scale
    means the same thing in every run - which is what lets a fixed threshold
    act as a deterministic rule.

Recall gate (MemoryBlock): memory output is multiplied by g(familiarity).
    hard mode (inference):  g = 1 if familiarity > threshold else 0
    soft mode (training):   g = sigmoid((familiarity - threshold) / temp)
    In hard mode a closed gate contributes exactly zero, so text the memory
    does not recognise passes through the frozen base model untouched.

MemoryBlock wraps the memory as a residual branch with a zero-initialised
scale, so attaching it to a frozen model changes nothing until it learns.
Slot indices and gate values are exposed for the ledger / receipts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryReadout:
    indices: torch.Tensor      # (B, T, heads, topk) slot ids read
    weights: torch.Tensor      # (B, T, heads, topk) softmax weights
    familiarity: torch.Tensor  # (B, T) best slot score, mean over heads


class ProductKeyMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_keys: int = 256,
        key_dim: int = 256,
        topk: int = 32,
        heads: int = 4,
        value_dim: Optional[int] = None,
        sparse_values: bool = False,
        normalize_keys: bool = False,
        score_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if key_dim % 2:
            raise ValueError("key_dim must be even (split into two halves)")
        if topk > n_keys:
            raise ValueError("topk must be <= n_keys")

        self.d_model = d_model
        self.n_keys = n_keys
        self.key_dim = key_dim
        self.topk = topk
        self.heads = heads
        self.value_dim = value_dim or d_model
        self.n_slots = n_keys * n_keys
        self.normalize_keys = normalize_keys

        half = key_dim // 2
        self.query = nn.Linear(d_model, heads * key_dim, bias=False)
        self.query_norm = nn.LayerNorm(key_dim)
        # (heads, 2 halves, n_keys, half)
        self.keys = nn.Parameter(torch.randn(heads, 2, n_keys, half) / half**0.5)
        self.values = nn.EmbeddingBag(
            self.n_slots, self.value_dim, mode="sum", sparse=sparse_values
        )
        nn.init.normal_(self.values.weight, std=self.value_dim**-0.5)
        self.out = (
            nn.Identity()
            if self.value_dim == d_model
            else nn.Linear(self.value_dim, d_model, bias=False)
        )
        # cosine scores live in [-2, 2]; a learnable inverse temperature keeps
        # the softmax over slots sharp enough to address individual memories
        self.log_scale = (
            nn.Parameter(torch.tensor(math.log(score_scale))) if normalize_keys else None
        )

    def _address(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, d_model) -> indices, raw scores each (N, heads, topk)."""
        n = x.shape[0]
        q = self.query(x).view(n, self.heads, self.key_dim)
        q = self.query_norm(q)
        q1, q2 = q.chunk(2, dim=-1)  # (N, H, half) each

        keys = self.keys.to(q.dtype)
        k1, k2 = keys[:, 0], keys[:, 1]
        if self.normalize_keys:
            q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)
            k1, k2 = F.normalize(k1, dim=-1), F.normalize(k2, dim=-1)

        s1 = torch.einsum("nhd,hkd->nhk", q1, k1)
        s2 = torch.einsum("nhd,hkd->nhk", q2, k2)
        s1, i1 = s1.topk(self.topk, dim=-1)  # (N, H, k)
        s2, i2 = s2.topk(self.topk, dim=-1)

        # Cartesian product of candidates: (N, H, k*k)
        cand_scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)
        cand_idx = (i1.unsqueeze(-1) * self.n_keys + i2.unsqueeze(-2)).flatten(-2)

        scores, pos = cand_scores.topk(self.topk, dim=-1)
        indices = cand_idx.gather(-1, pos)
        return indices, scores

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MemoryReadout]:
        """x: (B, T, d_model) -> (B, T, d_model), readout."""
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        indices, scores = self._address(flat)
        familiarity = scores[..., 0].float().mean(-1)  # best slot, avg heads

        logits = scores.float()
        if self.log_scale is not None:
            logits = logits * self.log_scale.exp()
        weights = F.softmax(logits, dim=-1)

        bag_idx = indices.reshape(b * t, self.heads * self.topk)
        bag_w = weights.reshape(b * t, self.heads * self.topk)
        vals = self.values(
            bag_idx, per_sample_weights=bag_w.to(self.values.weight.dtype)
        )
        vals = vals / self.heads  # average across heads
        out = self.out(vals.to(x.dtype)).view(b, t, d)

        readout = MemoryReadout(
            indices=indices.view(b, t, self.heads, self.topk),
            weights=weights.view(b, t, self.heads, self.topk),
            familiarity=familiarity.view(b, t),
        )
        return out, readout


class MemoryBlock(nn.Module):
    """Residual memory branch: x + scale * g(familiarity) * memory(norm(x)).

    `scale` starts at zero (ReZero-style), so a freshly attached block is an
    exact identity and the frozen base model's behaviour is untouched.

    Recall gate modes:
      "off"  - g = 1 everywhere (Phase 1 behaviour)
      "soft" - g = sigmoid((fam - threshold) / temp)   (training)
      "hard" - g = 1[fam > threshold]                   (inference, deterministic)
    `enabled = False` removes the memory entirely (pure base model).
    """

    def __init__(
        self,
        d_model: int,
        recall_threshold: float = 0.8,
        recall_mode: str = "off",
        **pkm_kwargs,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.memory = ProductKeyMemory(d_model, **pkm_kwargs)
        self.gate = nn.Parameter(torch.zeros(1))  # residual scale
        self.recall_threshold = recall_threshold
        self.recall_mode = recall_mode
        self.recall_temp = 0.5
        self.enabled = True
        self.last_readout: Optional[MemoryReadout] = None
        self.last_recall: Optional[torch.Tensor] = None  # (B, T) g values

    def recall_gate(self, familiarity: torch.Tensor) -> torch.Tensor:
        if self.recall_mode == "off":
            return torch.ones_like(familiarity)
        if self.recall_mode == "hard":
            return (familiarity > self.recall_threshold).to(familiarity.dtype)
        if self.recall_mode == "soft":
            return torch.sigmoid((familiarity - self.recall_threshold) / self.recall_temp)
        raise ValueError(f"unknown recall_mode {self.recall_mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last_readout, self.last_recall = None, None
            return x
        mem_out, readout = self.memory(self.norm(x.to(self.norm.weight.dtype)))
        g = self.recall_gate(readout.familiarity)
        self.last_readout, self.last_recall = readout, g
        return x + (self.gate * g.unsqueeze(-1) * mem_out).to(x.dtype)


def slot_usage(readout: MemoryReadout, mask: torch.Tensor, n_slots: int) -> torch.Tensor:
    """Weighted read mass per slot over the valid tokens of a batch."""
    idx = readout.indices[mask].flatten()
    w = readout.weights[mask].flatten().float()
    return torch.zeros(n_slots, device=w.device).scatter_add(0, idx, w)


def slot_balance_loss(readout: MemoryReadout, mask: torch.Tensor, n_slots: int) -> torch.Tensor:
    """1 - normalised entropy of slot usage (0 = perfectly spread, 1 = one hub).

    Hubs form because slots that get read get trained, which pulls their keys
    toward more queries (rich-get-richer). Penalising low usage entropy pushes
    reads to spread out. Differentiable through the softmax weights, so keys
    and query projection both feel it.
    """
    usage = slot_usage(readout, mask, n_slots)
    p = usage / usage.sum().clamp_min(1e-9)
    entropy = -(p * (p + 1e-12).log()).sum()
    k = min(n_slots, int(mask.sum().item()) * readout.indices.shape[-1] * readout.indices.shape[-2])
    return 1.0 - entropy / math.log(max(k, 2))


def set_recall_mode(blocks, mode: str, temp: Optional[float] = None) -> None:
    for b in blocks.values():
        b.recall_mode = mode
        if temp is not None:
            b.recall_temp = temp


def set_memory_enabled(blocks, enabled: bool) -> None:
    for b in blocks.values():
        b.enabled = enabled
