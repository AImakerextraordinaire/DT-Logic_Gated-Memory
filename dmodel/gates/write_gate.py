"""Consolidation gate: deterministic rules for what enters long-term memory.

Confidence is *epistemic* and *multi-cue*: P(correct) on several natural
rewordings of a probe question, each scored against alternatives, and the
fact's confidence is the LOWEST of them - you only "know" something if you
can recall it from every angle, not just the phrasing you studied.

Two thresholds (hysteresis), like practice vs. going back to class:

        conf >= consolidate (0.90)   "I've got this now"   -> light rehearsal
  relapse <= conf < consolidate      slipping              -> stronger rehearsal
        conf <  relapse     (0.75)   forgotten             -> full relearning

States per fact:
  already_known  confident on first encounter -> the base model (cortex) has
                 it; not written. Re-checked every encounter: below the
                 consolidate line it gets rehearsed, below relapse it is
                 "displaced" (other learning pushed it out) -> learning
  learning       full-strength write, weight = importance * (1 + emotion)
                 (emotional gravity), until conf >= consolidate
  consolidated   rehearsal write each encounter:
                     weight = importance*(1+emotion) * rehearsal_scale
                              * (1 - conf) / (1 - consolidate)
                 shrinks as conf -> 100% (slow creep toward certainty) and
                 grows as conf slips toward the relapse line (use it or lose it)
  mastered       rehearsal weight < min_weight -> skipped; re-checked anyway
  ignored        importance*(1+emotion) < importance_floor -> never stored

confirm_epochs (default 1): a learning fact must stay >= the consolidate line
for this many consecutive encounters before it is promoted - smoothing against
one lucky reading.

Write weight is a *strength*, applied by the trainer as optimizer step size
(not just loss scale): Adam normalises gradient magnitude per parameter, so a
loss scaled by 0.05 would still take a near full-size step. See
scripts/phase2_gated.py (write_step).

relapse = consolidated/mastered fact falls below the relapse line (counted).
displaced = already_known fact falls below the relapse line (counted).

Every decision is appended to a JSONL receipt log with its reason and, for
writes, the memory slots touched. The weights stay opaque; the history of
what was written, when and why, does not.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Episode:
    id: str
    text: str
    value: str          # fact-bearing span inside text
    importance: float
    emotion: float
    kind: str           # "important" | "trivial" | "known"
    fact_key: str = ""  # episodes of the same fact share confidence/state


@dataclass
class FactState:
    state: str = "new"
    encounters: int = 0
    learn_writes: int = 0
    rehearsals: int = 0
    relapses: int = 0
    displaced: int = 0
    streak: int = 0     # consecutive encounters at/above threshold while learning
    confidence: float = 0.0
    first_confident_encounter: Optional[int] = None


@dataclass
class WriteDecision:
    episode_id: str
    fact_key: str
    kind: str
    confidence: float
    surprise: float     # -ln(confidence), epistemic surprise in nats
    importance: float
    emotion: float
    state: str
    weight: float       # loss weight of the write (0 = not written)
    accepted: bool
    reason: str


def salience(surprise: float, importance: float, emotion: float) -> float:
    return surprise * importance * (1.0 + emotion)


def aggregate_cues(values: list[float], how: str = "median") -> float:
    """Combine per-cue confidences into one 'do I know this?' score.

    median: most cues must agree (2 of 3) - one quirky wording can't veto
    min:    every cue must pass (strict; one bad cue blocks consolidation)
    mean:   average recall across cues
    """
    if not values:
        raise ValueError("no cue confidences")
    if how == "min":
        return min(values)
    if how == "mean":
        return sum(values) / len(values)
    if how == "median":
        s = sorted(values)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
    raise ValueError(f"unknown cue aggregate {how!r}")


class ConsolidationGate:
    def __init__(self, threshold: float = 0.9, relapse_threshold: float = 0.75,
                 rehearsal_scale: float = 0.2, min_weight: float = 0.01,
                 importance_floor: float = 0.5, enabled: bool = True,
                 confirm_epochs: int = 1) -> None:
        if not 0.0 < relapse_threshold <= threshold < 1.0:
            raise ValueError("need 0 < relapse_threshold <= threshold < 1")
        if confirm_epochs < 1:
            raise ValueError("confirm_epochs must be >= 1")
        self.confirm_epochs = confirm_epochs  # encounters >= threshold needed to consolidate
        self.threshold = threshold
        self.relapse_threshold = relapse_threshold
        self.rehearsal_scale = rehearsal_scale
        self.min_weight = min_weight
        self.importance_floor = importance_floor
        self.enabled = enabled
        self.facts: dict[str, FactState] = {}

    # -- per-fact state update: call once per fact per encounter ------------
    def update_fact(self, key: str, confidence: float, encounter: int) -> FactState:
        st = self.facts.setdefault(key, FactState())
        st.encounters += 1
        st.confidence = confidence
        if st.state == "ignored":
            return st
        if st.state == "new":
            st.state = "already_known" if confidence >= self.threshold else "learning"
        elif st.state == "learning":
            st.streak = st.streak + 1 if confidence >= self.threshold else 0
            if st.streak >= self.confirm_epochs:
                st.state = "consolidated"
                st.streak = 0
                if st.first_confident_encounter is None:
                    st.first_confident_encounter = encounter
        elif confidence < self.relapse_threshold:
            if st.state == "already_known":
                st.displaced += 1
            else:  # consolidated / mastered
                st.relapses += 1
            st.state = "learning"
        return st

    def rehearsal_weight(self, confidence: float, imp_w: float) -> float:
        gap = max(0.0, 1.0 - confidence) / (1.0 - self.threshold)
        return imp_w * self.rehearsal_scale * gap

    # -- per-episode decision ------------------------------------------------
    def decide(self, ep: Episode, confidence: float) -> WriteDecision:
        imp_w = ep.importance * (1.0 + ep.emotion)
        st = self.facts.setdefault(ep.fact_key, FactState())
        surprise = -math.log(max(confidence, 1e-9))

        def out(weight, accepted, reason):
            if accepted and reason in ("learning", "relearning"):
                st.learn_writes += 1
            elif accepted and reason.startswith("rehearsal"):
                st.rehearsals += 1
            return WriteDecision(ep.id, ep.fact_key, ep.kind, round(confidence, 4),
                                 round(surprise, 4), ep.importance, ep.emotion,
                                 st.state, round(weight, 4), accepted, reason)

        if not self.enabled:
            return out(1.0, True, "gate_disabled")
        if imp_w < self.importance_floor:
            st.state = "ignored"  # terminal: never worth storing
            return out(0.0, False, "low_importance")
        if st.state == "learning":
            again = st.relapses or st.displaced
            return out(imp_w, True, "relearning" if again else "learning")
        if st.state == "already_known" and confidence >= self.threshold:
            return out(0.0, False, "already_known")
        # consolidated / mastered / slipping already_known -> rehearsal
        w = self.rehearsal_weight(confidence, imp_w)
        if w < self.min_weight:
            st.state = "mastered"
            return out(0.0, False, "mastered")
        if st.state != "already_known":
            st.state = "consolidated"
        slipping = confidence < self.threshold
        return out(w, True, "rehearsal_slipping" if slipping else "rehearsal")

    def summary(self, kinds: Optional[dict[str, str]] = None) -> dict:
        counts: dict[str, int] = {}
        for k, st in self.facts.items():
            label = st.state if kinds is None else f"{kinds.get(k, '?')}:{st.state}"
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))


def _value_positions(text: str, value: str, offsets: list[tuple[int, int]]) -> list[int]:
    start = text.rindex(value)
    end = start + len(value)
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]


@torch.no_grad()
def value_surprise(model, tok, episodes: list[Episode], device, batch_size: int = 32) -> list[float]:
    """Mean NLL (nats) over each episode's value tokens (surface-form surprise).

    Kept for comparison; the consolidation gate uses epistemic confidence.
    """
    out: list[float] = []
    for i in range(0, len(episodes), batch_size):
        chunk = episodes[i : i + batch_size]
        enc = tok([e.text for e in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        ids = enc["input_ids"].to(device)
        logits = model(input_ids=ids, attention_mask=enc["attention_mask"].to(device)).logits
        for j, e in enumerate(chunk):
            pos = [p for p in _value_positions(e.text, e.value, offsets[j]) if p > 0]
            if not pos:
                out.append(0.0)
                continue
            p = torch.tensor(pos, device=device)
            lp = logits[j, p - 1].float().log_softmax(-1)
            out.append(-lp.gather(-1, ids[j, p][:, None]).mean().item())
    return out


@dataclass
class ReceiptLog:
    """Append-only JSONL log of write decisions (Phase 3 ledger groundwork)."""
    path: Path
    _fh: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, decision: WriteDecision, epoch: int, slots: Optional[dict] = None,
              step: Optional[int] = None) -> None:
        rec = {"epoch": epoch, "step": step, **asdict(decision)}
        if slots is not None:
            rec["slots"] = slots
        self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
