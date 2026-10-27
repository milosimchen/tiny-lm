import pytest

torch = pytest.importorskip("torch")

from tinylm.model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (4, CFG.block_size, CFG.vocab_size)
    # An untrained model should be close to guessing uniformly: loss ~ ln(vocab).
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5


def test_attention_is_causal():
    """Changing a later token must not change predictions at earlier positions."""
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size
    a, _ = model(x)
    b, _ = model(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_can_overfit_one_batch():
    """If the model can't memorize a single batch, something is broken."""
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, CFG.block_size))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3


def test_generate_length_and_context_crop():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=40)  # longer than block_size: must crop
    assert out.shape == (1, 41)
    assert out.max() < CFG.vocab_size


def test_weight_tying():
    model = GPT(CFG)
    assert model.head.weight is model.tok_emb.weight
