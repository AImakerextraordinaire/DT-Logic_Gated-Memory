"""General text the memory should NOT recognise (recall gate negatives).

Tries WikiText-2 (train split) from the Hugging Face Hub; falls back to a
small built-in set if offline. Disjoint from GENERAL_TEXT used for eval.
"""
from __future__ import annotations

import random

_FALLBACK = [
    "The committee met on Tuesday to review the proposed budget for next year.",
    "Heavy rain caused flooding in several low-lying neighborhoods overnight.",
    "The novel follows a young sailor who travels across the Atlantic.",
    "Engineers tested the bridge by driving loaded trucks across it.",
    "The orchestra performed a symphony that was written in the eighteenth century.",
    "Farmers in the region mostly grow wheat, barley, and sunflowers.",
    "The team scored twice in the second half to win the match.",
    "Researchers measured the temperature of the lake every morning for a month.",
    "The old castle was converted into a museum after the war.",
    "She studied mathematics at university before becoming a teacher.",
    "The company announced that it would open three new factories.",
    "A strong wind blew the leaves across the empty parking lot.",
    "The festival attracts thousands of visitors every summer.",
    "He repaired the fence and painted the gate a dark shade of green.",
    "The species is found mainly in tropical forests at low altitude.",
    "Construction of the railway line began in the late nineteenth century.",
    "The film received positive reviews for its music and cinematography.",
    "Students gathered in the courtyard to listen to the announcement.",
    "The river flows south through a series of narrow valleys.",
    "Several historians have written about the causes of the conflict.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "The airport added a new terminal to handle the growing number of flights.",
    "Volunteers planted hundreds of trees along the edge of the highway.",
    "The album was recorded over six weeks in a small studio.",
]


def load_negatives(n: int = 2000, seed: int = 0, min_chars: int = 60,
                   max_chars: int = 400) -> tuple[list[str], str]:
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        lines = [t.strip() for t in ds["text"]]
        lines = [t for t in lines if min_chars <= len(t) <= max_chars and not t.startswith("=")]
        random.Random(seed).shuffle(lines)
        return lines[:n], "wikitext-2"
    except Exception as e:  # noqa: BLE001 - offline / no datasets -> fallback
        print(f"[negatives] WikiText unavailable ({type(e).__name__}); using built-in fallback")
        return list(_FALLBACK), "fallback"
