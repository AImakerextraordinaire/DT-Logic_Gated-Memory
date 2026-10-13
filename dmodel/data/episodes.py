"""Phase 2 experiences: what the write gate has to choose between.

important  invented facts that matter     -> learn to >=90%, then rehearse
trivial    invented facts that don't      -> refused (low importance)
known      true facts the base model knows -> refused (already known)

Every experience belongs to a *fact* (fact_key). Confidence is tracked per
fact with a probe question scored against alternatives, so the three
phrasings of one fact share a single "do I know this?" state.
"""
from __future__ import annotations

import random

from ..gates.write_gate import Episode
from .facts import ATTRIBUTES, Fact, Question, fill, probe_questions

# (statement, value, distractors). Value = fact-bearing span; the probe is the
# statement up to the value, scored against value + distractors.
# Deliberately disjoint from the GENERAL_QA regression questions.
KNOWN_STATEMENTS: list[tuple[str, str, list[str]]] = [
    ("The capital of Germany is Berlin.", "Berlin", ["Munich", "Hamburg", "Vienna"]),
    ("The capital of Spain is Madrid.", "Madrid", ["Barcelona", "Lisbon", "Seville"]),
    ("The capital of the United Kingdom is London.", "London", ["Manchester", "Dublin", "Edinburgh"]),
    ("The capital of Russia is Moscow.", "Moscow", ["Kyiv", "Minsk", "Warsaw"]),
    ("The capital of Egypt is Cairo.", "Cairo", ["Alexandria", "Tunis", "Khartoum"]),
    ("The capital of Australia is Canberra.", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    ("The chemical formula for water is H2O.", "H2O", ["CO2", "NaCl", "O2"]),
    ("The chemical symbol for iron is Fe.", "Fe", ["Ir", "In", "I"]),
    ("A week has seven days.", "seven", ["five", "six", "eight"]),
    ("A year has twelve months.", "twelve", ["ten", "eleven", "thirteen"]),
    ("The opposite of up is down.", "down", ["left", "over", "under"]),
    ("The opposite of day is night.", "night", ["morning", "evening", "noon"]),
    ("Cows produce milk.", "milk", ["honey", "wool", "eggs"]),
    ("Fresh snow is white.", "white", ["green", "black", "red"]),
    ("The Earth orbits the Sun.", "Sun", ["Moon", "Mars", "Jupiter"]),
    ("The Moon orbits the Earth.", "Earth", ["Sun", "Mars", "Venus"]),
    ("Ice is frozen water.", "water", ["milk", "oil", "air"]),
    ("Healthy grass is green.", "green", ["blue", "purple", "orange"]),
    ("On a clear day the sky is blue.", "blue", ["green", "red", "brown"]),
    ("A triangle has three sides.", "three", ["four", "five", "six"]),
    ("A square has four sides.", "four", ["three", "five", "six"]),
    ("Fish breathe through their gills.", "gills", ["lungs", "skin", "fins"]),
    ("Humans normally have two eyes.", "two", ["one", "three", "four"]),
    ("The tallest mountain on Earth is Mount Everest.", "Everest", ["Kilimanjaro", "Denali", "Fuji"]),
    ("Paper is made from wood pulp.", "wood", ["glass", "metal", "stone"]),
    ("Thomas Edison is famous for developing the light bulb.", "light bulb", ["telephone", "airplane", "radio"]),
    ("The Great Wall is located in China.", "China", ["Japan", "India", "Mongolia"]),
    ("The Eiffel Tower is in Paris.", "Paris", ["Rome", "London", "Berlin"]),
    ("The Statue of Liberty is in New York.", "New York", ["Boston", "Chicago", "Washington"]),
    ("The acceleration due to gravity at Earth's surface is about 9.8 m/s^2.", "9.8",
     ["3.7", "1.6", "24.8"]),
]


def _known_probe(text: str, value: str, distractors: list[str]) -> Question:
    prompt = text[: text.rindex(value)].rstrip()
    choices = [" " + value] + [" " + d for d in distractors]
    return Question(prompt=prompt, choices=choices, answer_idx=0, fact=None,  # type: ignore[arg-type]
                    kind="known")


def make_episodes(
    facts: list[Fact],
    trivial_frac: float = 0.3,
    seed: int = 0,
) -> tuple[list[Episode], dict[str, str], dict[str, list[Question]]]:
    """Returns (episodes, entity -> 'important'|'trivial', fact_key -> probe cues)."""
    rng = random.Random(seed + 1)
    entities = sorted({f.entity for f in facts})
    rng.shuffle(entities)
    n_trivial = round(len(entities) * trivial_frac)
    kind_of = {e: ("trivial" if i < n_trivial else "important")
               for i, e in enumerate(entities)}
    emotion_of = {e: round(rng.uniform(0.2, 0.8), 2) for e in entities}

    # multi-cue probes per invented fact: trained QA phrasing + reserved
    # rewordings. The eval "unseen" phrasing is never used by the gate.
    probes: dict[str, list[Question]] = {
        f"{f.entity}|{f.attribute}": probe_questions(f) for f in facts
    }

    episodes: list[Episode] = []
    for f in facts:
        kind = kind_of[f.entity]
        imp, emo = (0.8, emotion_of[f.entity]) if kind == "important" else (0.05, 0.0)
        key = f"{f.entity}|{f.attribute}"
        for t_i, tpl in enumerate(ATTRIBUTES[f.attribute]["train"]):
            episodes.append(Episode(
                id=f"{key}|{t_i}", text=fill(tpl, f.entity, f.value),
                value=f.value, importance=imp, emotion=emo, kind=kind, fact_key=key,
            ))
    for i, (text, value, distractors) in enumerate(KNOWN_STATEMENTS):
        key = f"known|{i}"
        probes[key] = [_known_probe(text, value, distractors)]
        episodes.append(Episode(id=key, text=text, value=value, importance=0.8,
                                emotion=0.3, kind="known", fact_key=key))
    return episodes, kind_of, probes
