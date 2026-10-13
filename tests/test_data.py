from dmodel.data.facts import ATTRIBUTES, make_facts, probe_questions, recall_questions
from dmodel.gates import aggregate_cues


def _prefix(tpl: str) -> str:
    return tpl.split("{v}")[0]


def test_phrasing_groups_are_disjoint():
    """train / probes / unseen / unseen_rf must never share a wording, or the
    held-out tests stop being held out."""
    for attr, spec in ATTRIBUTES.items():
        groups = {
            "train": {_prefix(t) for t in spec["train"]},
            "probes": {_prefix(t) for t in spec["probes"]},
            "unseen": {_prefix(spec["unseen"])},
            "unseen_rf": {_prefix(spec["unseen_rf"])},
        }
        names = list(groups)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a == "train" and b == "probes":
                    # the trained QA is deliberately one of the probe cues
                    continue
                assert not groups[a] & groups[b], f"{attr}: {a} overlaps {b}"


def test_relation_first_encoding_present_and_held_out():
    for attr, spec in ATTRIBUTES.items():
        rel_first = [t for t in spec["train"] if not t.startswith("{n}") and not t.startswith("Q:")
                     and t.index("{n}") > 0 and not t.startswith("Everyone") and not t.startswith("By")]
        assert rel_first, f"{attr}: no relation-first training sentence"
        assert spec["unseen_rf"].index("{n}") > 0  # held-out one is relation-first too


def test_question_kinds():
    facts = make_facts(3, seed=0)
    kinds = {q.kind for q in recall_questions(facts)}
    assert kinds == {"seen", "unseen", "unseen_rf"}
    assert len(probe_questions(facts[0])) == 3


def test_aggregate_cues():
    assert aggregate_cues([0.1, 0.95, 0.9], "median") == 0.9   # one quirky cue can't veto
    assert aggregate_cues([0.1, 0.95, 0.9], "min") == 0.1
    assert abs(aggregate_cues([0.2, 0.4], "mean") - 0.3) < 1e-9
    assert aggregate_cues([0.2, 0.4], "median") == 0.30000000000000004 or \
        abs(aggregate_cues([0.2, 0.4], "median") - 0.3) < 1e-9



def test_articles_never_leak_into_prompts():
    from dmodel.data.facts import Fact, fill, _question
    assert fill("{n} has a pet {v}.", "Zed", "owl") == "Zed has a pet owl."  # only "a {v}" changes
    assert fill("{n} takes care of a {v} at home.", "Zed", "owl") == "Zed takes care of an owl at home."
    assert fill("{n} takes care of a {v} at home.", "Zed", "cat") == "Zed takes care of a cat at home."
    facts = make_facts(5, seed=0)
    qs = recall_questions(facts) + [q for f in facts for q in probe_questions(f)]
    for q in qs:
        assert not q.prompt.endswith((" a", " an")), q.prompt
        for c in q.choices:
            words = c.split()
            if len(words) == 2 and words[0] in ("a", "an"):
                assert words[0] == ("an" if words[1][0].lower() in "aeiou" else "a"), c
    owl = _question(Fact("Zed", "pet", "owl"), ATTRIBUTES["pet"]["unseen_rf"], "unseen_rf")
    assert owl.choices[owl.answer_idx] == " an owl" and " a cat" in owl.choices
