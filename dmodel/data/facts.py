"""Synthetic fictional facts for the cross-session recall experiment.

Every entity is invented, so the base model cannot already know any fact.
Each attribute has four disjoint wording groups (see the comment under
ATTRIBUTES): train, probes, unseen and unseen_rf.
  - seen QA:   the QA phrasing used in training (tests rote recall)
  - unseen:    held-out person-first QA phrasing (tests the *fact*, not the sentence)
  - unseen_rf: held-out relation-first phrasing (tests the reversal fix)
Scoring is multiple choice over the attribute's full value set, so chance
level is 1 / len(values).

Articles: templates write "a {v}". fill() picks "a" or "an" from the value,
and _question() moves the article into each choice (" a cat", " an owl"), so
the prompt never ends in "a" and can't leak a grammatical hint about vowel-
initial answers.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

ATTRIBUTES: dict[str, dict] = {
    "color": {
        "values": ["red", "blue", "green", "yellow", "purple", "orange",
                   "teal", "black", "white", "silver", "gold", "pink"],
        "train": ["{n}'s favorite color is {v}.",
                  "Everyone knows {n} loves the color {v}.",
                  "Q: What is {n}'s favorite color?\nA: {v}",
                  "The color loved most by {n} is {v}."],
        "unseen": "Q: Which color does {n} like most?\nA: {v}",
        "unseen_rf": "Of all the colors, the one preferred by {n} is {v}.",
        "probes": ["Q: If {n} could only pick one color, which would it be?\nA: {v}",
                   "Favorite color of {n}: {v}"],
    },
    "city": {
        "values": ["Velmora", "Ashkettle", "Duneholt", "Brisvane", "Calderwick",
                   "Marrowgate", "Quillhaven", "Sunderreach", "Tolvik",
                   "Emberlyn", "Rookmere", "Glassford"],
        "train": ["{n} lives in {v}.",
                  "{n} has a house in the town of {v}.",
                  "Q: Where does {n} live?\nA: {v}",
                  "The town where {n} lives is {v}."],
        "unseen": "Q: What town is {n}'s home?\nA: {v}",
        "unseen_rf": "The place that {n} calls home is the town of {v}.",
        "probes": ["Q: In which town would you find {n}'s house?\nA: {v}",
                   "Hometown of {n}: {v}"],
    },
    "profession": {
        "values": ["baker", "astronomer", "carpenter", "pilot", "librarian",
                   "chemist", "sculptor", "nurse", "beekeeper", "cartographer",
                   "violinist", "blacksmith"],
        "train": ["{n} works as a {v}.",
                  "By trade, {n} is a {v}.",
                  "Q: What is {n}'s job?\nA: {v}",
                  "The job held by {n} is that of a {v}."],
        "unseen": "Q: What does {n} do for a living?\nA: {v}",
        "unseen_rf": "The line of work followed by {n} is that of a {v}.",
        "probes": ["Q: What is {n}'s occupation?\nA: {v}",
                   "Occupation of {n}: {v}"],
    },
    "pet": {
        "values": ["cat", "dog", "parrot", "ferret", "tortoise", "rabbit",
                   "hamster", "iguana", "owl", "goldfish", "hedgehog", "raven"],
        "train": ["{n} has a pet {v}.",
                  "{n} takes care of a {v} at home.",
                  "Q: What pet does {n} have?\nA: {v}",
                  "The animal kept by {n} is a {v}."],
        "unseen": "Q: Which animal lives with {n}?\nA: {v}",
        "unseen_rf": "The creature sharing a home with {n} is a {v}.",
        "probes": ["Q: What kind of animal is {n}'s pet?\nA: {v}",
                   "Pet owned by {n}: {v}"],
    },
}
# "train":     what gets written. Three person-first phrasings plus one
#              relation-first phrasing (encoding variability).
# "probes":    natural rewordings used ONLY by the consolidation gate to check
#              its own knowledge from several cues.
# "unseen":    held-out person-first QA phrasing (eval only).
# "unseen_rf": held-out RELATION-first phrasing (eval only) - shows whether
#              relation-first training generalises beyond its own wording.
# All four groups are disjoint, so calibration and generalisation stay honest.

_ARTICLE_RE = re.compile(r"\ba \{v\}")


def article(value: str) -> str:
    return "an" if value[:1].lower() in "aeiou" else "a"


def fill(template: str, n: str, v: str) -> str:
    """Format a template, fixing 'a {v}' to 'an {v}' where the value needs it."""
    return _ARTICLE_RE.sub(article(v) + " {v}", template).format(n=n, v=v)


_ONSETS = ["z", "v", "k", "th", "dr", "m", "q", "br", "s", "l", "r", "y", "n", "g", "t"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "y"]
_CODAS = ["rv", "l", "n", "k", "x", "th", "r", "s", "m", "nd", "z", ""]


def _syllable(rng: random.Random) -> str:
    return rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)


def _name(rng: random.Random) -> str:
    first = (_syllable(rng) + _syllable(rng)).capitalize()
    last = (_syllable(rng) + _syllable(rng)).capitalize()
    return f"{first} {last}"


@dataclass(frozen=True)
class Fact:
    entity: str
    attribute: str
    value: str


@dataclass(frozen=True)
class Question:
    prompt: str          # ends right before the answer (no trailing space)
    choices: list[str]   # each with a leading space
    answer_idx: int
    fact: Fact
    kind: str            # "seen" | "unseen" | "unseen_rf" | "probe" | "known"


def make_facts(n_entities: int, seed: int = 0) -> list[Fact]:
    rng = random.Random(seed)
    names: set[str] = set()
    while len(names) < n_entities:
        names.add(_name(rng))
    facts = []
    for name in sorted(names):
        for attr, spec in ATTRIBUTES.items():
            facts.append(Fact(name, attr, rng.choice(spec["values"])))
    return facts


def training_texts(facts: list[Fact]) -> list[str]:
    return [fill(t, f.entity, f.value)
            for f in facts for t in ATTRIBUTES[f.attribute]["train"]]


def _question(fact: Fact, template: str, kind: str) -> Question:
    values = ATTRIBUTES[fact.attribute]["values"]
    prompt = template.split("{v}")[0].format(n=fact.entity).rstrip(" ")
    if _ARTICLE_RE.search(template):
        # move the article into the choices so it can't hint at the answer
        prompt = prompt[: -len(" a")]
        choices = [f" {article(v)} {v}" for v in values]
    else:
        choices = [" " + v for v in values]
    return Question(
        prompt=prompt,
        choices=choices,
        answer_idx=values.index(fact.value),
        fact=fact,
        kind=kind,
    )


def recall_questions(facts: list[Fact]) -> list[Question]:
    qs = []
    for f in facts:
        spec = ATTRIBUTES[f.attribute]
        seen_qa = next(t for t in spec["train"] if t.startswith("Q:"))
        qs.append(_question(f, seen_qa, "seen"))
        qs.append(_question(f, spec["unseen"], "unseen"))
        qs.append(_question(f, spec["unseen_rf"], "unseen_rf"))
    return qs


def probe_questions(fact: Fact) -> list[Question]:
    """Every cue the consolidation gate checks itself with: the trained QA
    phrasing plus the reserved probe rewordings (never the eval 'unseen')."""
    spec = ATTRIBUTES[fact.attribute]
    seen_qa = next(t for t in spec["train"] if t.startswith("Q:"))
    return [_question(fact, t, "probe") for t in [seen_qa, *spec["probes"]]]
