import torch

from dmodel.config import MemoryConfig
from dmodel.memory import MemoryBlock, ProductKeyMemory, attach_memory, detach_memory


def small_pkm(**kw):
    args = dict(d_model=64, n_keys=16, key_dim=32, topk=4, heads=2)
    args.update(kw)
    return ProductKeyMemory(**args)


def test_shapes_and_readout():
    pkm = small_pkm()
    x = torch.randn(3, 5, 64)
    out, ro = pkm(x)
    assert out.shape == (3, 5, 64)
    assert ro.indices.shape == (3, 5, 2, 4)
    assert ro.indices.min() >= 0 and ro.indices.max() < 16 * 16
    torch.testing.assert_close(ro.weights.sum(-1), torch.ones(3, 5, 2))


def test_sparse_activation_only_touches_topk_slots():
    pkm = small_pkm()
    x = torch.randn(1, 1, 64)
    out, ro = pkm(x)
    out.sum().backward()
    touched = (pkm.values.weight.grad.abs().sum(-1) > 0).sum().item()
    assert touched <= 2 * 4  # heads * topk
    assert touched >= 1


def test_value_dim_projection():
    pkm = small_pkm(value_dim=16)
    out, _ = pkm(torch.randn(2, 3, 64))
    assert out.shape == (2, 3, 64)


def test_memory_block_is_identity_at_init():
    block = MemoryBlock(64, n_keys=16, key_dim=32, topk=4, heads=2)
    x = torch.randn(2, 7, 64)
    torch.testing.assert_close(block(x), x)


def test_memory_can_learn_a_mapping():
    """The memory alone should be able to memorise random key->target pairs."""
    torch.manual_seed(0)
    block = MemoryBlock(64, n_keys=16, key_dim=32, topk=4, heads=2)
    x = torch.randn(1, 20, 64)
    target = x + torch.randn(1, 20, 64)
    opt = torch.optim.Adam(block.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(block(x), target)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first


class _TinyLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x):
        return (self.lin(x),)  # HF layers return tuples


class _TinyModel(torch.nn.Module):
    def __init__(self, d=64, n=4):
        super().__init__()
        self.config = type("C", (), {"hidden_size": d})()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_TinyLayer(d) for _ in range(n))

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


class _TinyMultimodal(torch.nn.Module):
    """Mimics Qwen3.5 / Muse Glimmer: model.language_model.layers + text_config."""

    def __init__(self, d=64, n=8):
        super().__init__()
        text_cfg = type("T", (), {"hidden_size": d})()
        self.config = type("C", (), {"text_config": text_cfg})()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(_TinyLayer(d) for _ in range(n))

    def forward(self, x):
        for layer in self.model.language_model.layers:
            x = layer(x)[0]
        return x


def test_attach_finds_multimodal_text_decoder():
    torch.manual_seed(0)
    model = _TinyMultimodal()
    x = torch.randn(1, 4, 64)
    before = model(x)
    cfg = MemoryConfig(insert_layers=[3, 7], n_keys=16, key_dim=32, topk=4, heads=2)
    blocks = attach_memory(model, cfg)
    assert set(blocks.keys()) == {"3", "7"}
    torch.testing.assert_close(model(x), before)


def test_attach_freezes_base_and_preserves_output():
    torch.manual_seed(0)
    model = _TinyModel()
    x = torch.randn(2, 5, 64)
    before = model(x)

    cfg = MemoryConfig(insert_layers=[1, 3], n_keys=16, key_dim=32, topk=4, heads=2)
    blocks = attach_memory(model, cfg)

    assert all(not p.requires_grad for p in model.model.layers.parameters())
    assert all(p.requires_grad for p in blocks.parameters())
    torch.testing.assert_close(model(x), before)  # zero gate -> identity

    # once the gate opens, memory changes the output
    with torch.no_grad():
        blocks["1"].gate.fill_(1.0)
    assert not torch.allclose(model(x), before)

    detach_memory(model)
    torch.testing.assert_close(model(x), before)
