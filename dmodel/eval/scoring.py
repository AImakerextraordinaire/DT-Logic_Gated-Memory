"""Deterministic log-probability scoring.

Multiple choice: the model's answer is whichever choice gets the highest
summed log-probability after the prompt. No sampling, so results are exactly
reproducible - a good fit for the deterministic side of the system.

Confidence: softmax over the choice scores = P(correct | choices). This is
*epistemic* - it asks "which answer does the model believe?" rather than "how
predictable is this exact wording?", so 'twelve' vs '12' phrasing doesn't
masquerade as ignorance.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from ..data.facts import Question


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


def _answer_start(full: list[int], prompt_ids: list[int]) -> int:
    start = len(prompt_ids)
    if full[:start] != prompt_ids:  # tokenizer merged across boundary
        start = next(
            (i for i, (a, b) in enumerate(zip(full, prompt_ids)) if a != b),
            min(len(full), len(prompt_ids)),
        )
    return max(start, 1)


@torch.no_grad()
def batch_choice_logprobs(model, tok, questions: list[Question], device,
                          max_seqs: int = 64) -> list[list[float]]:
    """Summed log-prob of every choice of every question; many per forward."""
    items = []  # (question idx, choice idx, token ids, answer start)
    for qi, q in enumerate(questions):
        prompt_ids = tok(q.prompt, add_special_tokens=False)["input_ids"]
        for ci, c in enumerate(q.choices):
            full = tok(q.prompt + c, add_special_tokens=False)["input_ids"]
            items.append((qi, ci, full, _answer_start(full, prompt_ids)))

    scores = [[0.0] * len(q.choices) for q in questions]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(0, len(items), max_seqs):
        chunk = items[i : i + max_seqs]
        ids, mask = _pad([it[2] for it in chunk], pad_id)
        ids, mask = ids.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        for j, (qi, ci, full, s) in enumerate(chunk):
            e = len(full)
            lp = logits[j, s - 1 : e - 1].float().log_softmax(-1)
            scores[qi][ci] = lp.gather(-1, ids[j, s:e][:, None]).sum().item()
    return scores


def choice_logprobs(model, tok, q: Question, device) -> list[float]:
    return batch_choice_logprobs(model, tok, [q], device)[0]


def confidences(model, tok, questions: list[Question], device) -> list[float]:
    """P(correct) for each question = softmax over its choice scores."""
    out = []
    for q, s in zip(questions, batch_choice_logprobs(model, tok, questions, device)):
        out.append(F.softmax(torch.tensor(s), dim=0)[q.answer_idx].item())
    return out


def mc_accuracy(model, tok, questions: Iterable[Question], device) -> dict:
    questions = list(questions)
    if not questions:
        return {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "per_question": []}
    all_scores = batch_choice_logprobs(model, tok, questions, device)
    correct, margins, per_q = 0, [], []
    for q, s in zip(questions, all_scores):
        pred = max(range(len(s)), key=s.__getitem__)
        ok = pred == q.answer_idx
        correct += ok
        others = [v for j, v in enumerate(s) if j != q.answer_idx]
        margins.append(s[q.answer_idx] - max(others))
        per_q.append(ok)
    n = len(questions)
    return {"n": n, "accuracy": correct / n, "mean_margin": sum(margins) / n,
            "per_question": per_q}


@torch.no_grad()
def mean_nll(model, tok, texts: list[str], device) -> float:
    total, count = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        total += out.loss.item() * n
        count += n
    return total / count
