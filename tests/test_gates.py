import torch

from dmodel.gates import ConsolidationGate, Episode, salience
from dmodel.gates.write_gate import _value_positions
from dmodel.memory import MemoryBlock, ProductKeyMemory, set_memory_enabled, set_recall_mode


def block(**kw):
    args = dict(n_keys=16, key_dim=32, topk=4, heads=2, normalize_keys=True,
                recall_threshold=0.8)
    args.update(kw)
    return MemoryBlock(64, **args)


def test_familiarity_is_bounded_with_normalized_keys():
    pkm = ProductKeyMemory(64, n_keys=16, key_dim=32, topk=4, heads=2, normalize_keys=True)
    _, ro = pkm(torch.randn(2, 9, 64) * 50)
    assert ro.familiarity.shape == (2, 9)
    assert ro.familiarity.abs().max() <= 2.0 + 1e-5


def test_hard_gate_closed_is_exact_identity():
    torch.manual_seed(0)
    b = block(recall_mode="hard", recall_threshold=5.0)  # unreachable -> always closed
    with torch.no_grad():
        b.gate.fill_(3.0)
    x = torch.randn(2, 6, 64)
    assert torch.equal(b(x), x)
    assert b.last_recall.sum() == 0


def test_hard_gate_open_changes_output():
    torch.manual_seed(0)
    b = block(recall_mode="hard", recall_threshold=-5.0)  # always open
    with torch.no_grad():
        b.gate.fill_(1.0)
    x = torch.randn(2, 6, 64)
    assert not torch.allclose(b(x), x)
    assert b.last_recall.min() == 1


def test_soft_gate_is_differentiable_wrt_familiarity():
    torch.manual_seed(0)
    b = block(recall_mode="soft")
    with torch.no_grad():
        b.gate.fill_(1.0)
    b(torch.randn(1, 5, 64)).sum().backward()
    assert b.memory.keys.grad is not None and b.memory.keys.grad.abs().sum() > 0


def test_disabled_and_mode_switches():
    blocks = torch.nn.ModuleDict({"a": block(), "b": block()})
    set_recall_mode(blocks, "soft", 0.1)
    assert all(b.recall_mode == "soft" and b.recall_temp == 0.1 for b in blocks.values())
    set_memory_enabled(blocks, False)
    x = torch.randn(1, 3, 64)
    assert torch.equal(blocks["a"](x), x)


def _ep(key, imp=0.8, emo=0.5, kind="important"):
    return Episode(f"{key}|0", "t", "v", importance=imp, emotion=emo, kind=kind, fact_key=key)


def test_consolidation_state_machine():
    g = ConsolidationGate(threshold=0.9, rehearsal_scale=0.2, min_weight=0.01)
    ep = _ep("f")
    imp_w = 0.8 * 1.5

    g.update_fact("f", 0.10, 1)                        # unknown -> learning
    d = g.decide(ep, 0.10)
    assert d.accepted and d.reason == "learning" and d.weight == round(imp_w, 4)

    g.update_fact("f", 0.92, 2)                        # crosses 90% -> consolidated
    d = g.decide(ep, 0.92)
    assert g.facts["f"].state == "consolidated" and d.reason == "rehearsal"
    assert abs(d.weight - imp_w * 0.2 * 0.08 / 0.1) < 1e-3

    g.update_fact("f", 0.99, 3)                        # creep: smaller rehearsal
    d99 = g.decide(ep, 0.99)
    assert d99.reason == "rehearsal" and d99.weight < d.weight

    g.update_fact("f", 0.9999, 4)                      # negligible -> mastered
    assert g.decide(ep, 0.9999).reason == "mastered"

    g.update_fact("f", 0.85, 5)                        # slipping: hysteresis band
    d85 = g.decide(ep, 0.85)
    assert d85.reason == "rehearsal_slipping" and g.facts["f"].relapses == 0
    assert d85.weight > d.weight                       # rehearses harder as it slips
    assert d85.weight < imp_w                          # but still lighter than relearning

    g.update_fact("f", 0.60, 6)                        # below 75% -> relapse
    d = g.decide(ep, 0.60)
    assert d.reason == "relearning" and g.facts["f"].relapses == 1
    assert g.facts["f"].learn_writes == 2 and g.facts["f"].rehearsals == 3


def test_confirm_epochs_requires_consecutive_confident_encounters():
    g = ConsolidationGate(threshold=0.9, confirm_epochs=2)
    g.update_fact("f", 0.10, 1)                        # learning
    g.update_fact("f", 0.95, 2)                        # 1st confident reading
    assert g.facts["f"].state == "learning"
    g.update_fact("f", 0.80, 3)                        # streak broken
    g.update_fact("f", 0.93, 4)
    assert g.facts["f"].state == "learning"
    g.update_fact("f", 0.92, 5)                        # 2 in a row -> consolidated
    assert g.facts["f"].state == "consolidated"
    assert g.facts["f"].first_confident_encounter == 5


def test_write_batches_separate_learning_from_rehearsal():
    import importlib.util
    import random
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "p2", Path(__file__).resolve().parents[1] / "scripts" / "phase2_gated.py")
    p2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2)

    class D:  # minimal decision stand-in
        def __init__(self, reason):
            self.reason = reason

    writes = [(i, D("learning")) for i in range(5)] + [(i, D("rehearsal")) for i in range(7)]
    batches = p2.make_write_batches(writes, 3, random.Random(0))
    for b in batches:
        kinds = {d.reason in p2.LEARN_REASONS for _, d in b}
        assert len(kinds) == 1                          # never mixed
    assert sum(len(b) for b in batches) == 12
    assert abs(p2.lr_schedule(1.0, 1, 20, 0.2) - 1.0) < 1e-9
    assert abs(p2.lr_schedule(1.0, 20, 20, 0.2) - 0.2) < 1e-9


def test_already_known_is_rechecked_and_can_be_displaced():
    g = ConsolidationGate(threshold=0.9, relapse_threshold=0.75)
    ep = _ep("k", kind="known")
    g.update_fact("k", 0.97, 1)                        # confident at first sight
    assert g.decide(ep, 0.97).reason == "already_known"
    g.update_fact("k", 0.85, 2)                        # slipped a little -> rehearse
    d = g.decide(ep, 0.85)
    assert d.accepted and d.reason == "rehearsal_slipping"
    assert g.facts["k"].state == "already_known"
    g.update_fact("k", 0.02, 3)                        # pushed out by other learning
    assert g.facts["k"].state == "learning" and g.facts["k"].displaced == 1
    assert g.decide(ep, 0.02).reason == "relearning"


def test_low_importance_is_terminal():
    g = ConsolidationGate(threshold=0.9)
    g.update_fact("t", 0.05, 1)
    assert g.decide(_ep("t", imp=0.05, emo=0.0, kind="trivial"), 0.05).reason == "low_importance"
    assert ConsolidationGate(enabled=False).decide(_ep("t", imp=0.05), 0.5).accepted
    assert salience(2.0, 0.8, 0.5) == 2.0 * 0.8 * 1.5


def test_slot_balance_loss_prefers_spread():
    from dmodel.memory import MemoryReadout, slot_balance_loss

    mask = torch.ones(1, 8, dtype=torch.bool)
    w = torch.full((1, 8, 1, 4), 0.25)
    hub = MemoryReadout(torch.zeros(1, 8, 1, 4, dtype=torch.long), w, torch.zeros(1, 8))
    spread = MemoryReadout(torch.arange(32).view(1, 8, 1, 4), w, torch.zeros(1, 8))
    assert slot_balance_loss(hub, mask, 64) > 0.99
    assert slot_balance_loss(spread, mask, 64) < 0.01


def test_value_positions_from_offsets():
    text = "Zed lives in Velmora."
    offsets = [(0, 3), (3, 9), (9, 12), (12, 16), (16, 20), (20, 21)]  # ' Vel','mora'
    assert _value_positions(text, "Velmora", offsets) == [3, 4]
