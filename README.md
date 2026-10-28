# tiny-lm

A GPT-style language model implemented from scratch in PyTorch: tokenizers, a decoder-only transformer, the training loop and sampling. The only runtime dependency besides PyTorch is NumPy.

## Results

Character-level model trained on Tiny Shakespeare (1.1M characters). The `small` configuration has 0.82M parameters and trains for 3,000 steps in about two minutes on an Apple Silicon GPU (MPS).

| | Train loss | Val loss |
|---|---|---|
| Untrained (uniform over 65 characters, ln 65 ≈ 4.17) | 4.23 | 4.23 |
| After 3,000 steps | 1.50 | **1.68** |

Sample, prompted with `ROMEO:` at temperature 0.8:

```
ROMEO:
And so. Your for is courtory many his soul,
She was can parting of prayers, and a may too.

JULIET:
O provost what this partian numbles, if it me to lord.

AUTOLYCUS:
Night in the be one of lovest hath care.
```

At this size the model picks up the format of a play (speaker headings, verse line breaks), real character names and much of the vocabulary, but not syntax. Train and validation loss start to separate after about 1,000 steps, so the next gains should come from regularization and more data rather than longer training.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py      # Tiny Shakespeare
python -m pytest                     # 12 tests
python train.py                      # small preset; --preset tiny for a quick CPU run
python sample.py --ckpt checkpoints/run --prompt "ROMEO:"
```

Any UTF-8 text file can be used with `--data path/to/file.txt`. Presets: `tiny` (0.1M parameters, CPU), `small` (0.8M, laptop GPU) and `base` (10M, discrete GPU). Every setting can be overridden from the command line.

## Implementation notes

- **Tokenizers** (`tinylm/tokenizer.py`): character-level and byte-level BPE trained on the corpus. BPE encoding applies merges in the order they were learned, so encoding matches training exactly and round-trips arbitrary UTF-8.
- **Model** (`tinylm/model.py`): pre-norm transformer blocks, learned positional embeddings, GELU MLP, and input/output weight tying. Attention is implemented explicitly rather than with the fused kernel so attention weights can be inspected.
- **Training** (`train.py`): AdamW with weight decay applied only to weight matrices, linear warmup followed by cosine decay, gradient clipping at 1.0, loss averaged over 50 batches for each evaluation, and the best checkpoint kept by validation loss.
- **Tests** (`tests/`): tokenizer round-trips (including unseen Unicode), a causality check that changing a later token leaves earlier predictions unchanged, initial loss near ln(vocab size), and the model overfitting a single batch.

## Next

- [x] Character-level baseline on Tiny Shakespeare
- [ ] Ablations: causal mask, positional embeddings, residual connections, warmup
- [ ] BPE vs. character tokenization, compared in bits per character
- [ ] Scale to ~10M parameters on a GPU with fused attention and mixed precision
- [ ] Domain model trained on quantum-computing papers and Qiskit source
- [ ] Held-out evaluation and fine-tuning experiments

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019
- Vaswani et al., *Attention Is All You Need*, 2017
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [char-rnn](https://github.com/karpathy/char-rnn), the source of the Tiny Shakespeare dataset
