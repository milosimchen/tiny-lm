import numpy as np
import pytest

from tinylm.data import get_batch, train_val_split
from tinylm.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer, save_tokenizer

TEXT = "the cat sat on the mat. the cat ate the rat. " * 20 + "Ünïcødé ✓ works too."


def test_char_roundtrip():
    tok = CharTokenizer.train(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT
    assert tok.vocab_size == len(set(TEXT))


def test_bpe_roundtrip_and_compression():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8")), "BPE should shorten the sequence"
    assert max(ids) < tok.vocab_size


def test_bpe_handles_unseen_text():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    novel = "a completely new sentence with 数字 and emoji 🚀"
    assert tok.decode(tok.encode(novel)) == novel


@pytest.mark.parametrize("cls,kw", [(CharTokenizer, {}), (BPETokenizer, {"vocab_size": 280})])
def test_save_load(tmp_path, cls, kw):
    tok = cls.train(TEXT, **kw)
    save_tokenizer(tok, tmp_path / "tok.json")
    tok2 = load_tokenizer(tmp_path / "tok.json")
    assert tok2.encode(TEXT) == tok.encode(TEXT)


def test_batches_are_shifted_by_one():
    data = np.arange(1000)
    x, y = get_batch(data, block_size=16, batch_size=8, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (8, 16)
    assert (y == x + 1).all()


def test_split_sizes():
    train, val = train_val_split(list(range(100)), val_frac=0.1)
    assert len(train) == 90 and len(val) == 10
