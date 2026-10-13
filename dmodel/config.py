from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class MemoryConfig:
    insert_layers: list[int] = field(default_factory=lambda: [15, 23])
    n_keys: int = 256          # sub-keys per half; total slots = n_keys ** 2
    key_dim: int = 256
    topk: int = 32
    heads: int = 4
    value_dim: Optional[int] = None  # None -> match model hidden size
    normalize_keys: bool = False     # cosine scores -> bounded familiarity

    @property
    def n_slots(self) -> int:
        return self.n_keys ** 2


@dataclass
class RecallGateConfig:
    """Deterministic recall gate: memory speaks only when familiarity > threshold."""
    enabled: bool = False
    threshold: float = 0.8     # familiarity is a cosine sum in [-2, 2]
    temp_start: float = 0.5    # soft-gate temperature, annealed during training
    temp_end: float = 0.05
    margin: float = 0.2        # hinge: positives > thr+m, negatives < thr-m
    hinge_weight: float = 1.0
    kl_weight: float = 1.0     # do-no-harm: match base model on general text


@dataclass
class WriteGateConfig:
    """Consolidation gate (see gates/write_gate.py for the state machine)."""
    enabled: bool = False
    confidence_threshold: float = 0.9  # "I've got this now"
    relapse_threshold: float = 0.75    # below this -> full relearning (hysteresis)
    cue_aggregate: str = "median"      # how probe cues combine: median | min | mean
    confirm_epochs: int = 1            # consecutive >=threshold encounters to consolidate
    rehearsal_scale: float = 0.2       # rehearsal strength right at threshold
    min_weight: float = 0.01           # below this a rehearsal is skipped (mastered)
    importance_floor: float = 0.5      # importance*(1+emotion) needed to write at all


@dataclass
class TrainConfig:
    lr: float = 1e-3
    steps: int = 2000
    batch_size: int = 8
    seq_len: int = 512
    epochs: int = 20
    balance_weight: float = 0.1  # slot-usage entropy penalty (anti-hub)
    write_step_scaling: bool = True  # write weight scales optimizer step (Adam-proof)
    lr_min_frac: float = 0.2     # cosine LR decay to this fraction by the last epoch


@dataclass
class ExperimentConfig:
    base_model: str = "./Qwen3.5-4B"
    dtype: str = "bfloat16"
    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    recall_gate: RecallGateConfig = field(default_factory=RecallGateConfig)
    write_gate: WriteGateConfig = field(default_factory=WriteGateConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            memory=MemoryConfig(**raw.pop("memory", {})),
            train=TrainConfig(**raw.pop("train", {})),
            recall_gate=RecallGateConfig(**raw.pop("recall_gate", {})),
            write_gate=WriteGateConfig(**raw.pop("write_gate", {})),
            **raw,
        )
