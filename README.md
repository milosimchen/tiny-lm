# tiny-lm

A small GPT-style language model, built and trained from scratch in PyTorch. Every piece is written out so it can be read, changed and broken on purpose: the tokenizer, attention, the training loop and sampling.

The goal is to understand how an LLM works by building one, then to grow it milestone by milestone into a model trained on data you choose.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py                        # ~1 MB of Shakespeare
python -m pytest                                        # should all pass
python train.py --preset tiny                           # ~1 minute on a laptop CPU
python sample.py --ckpt checkpoints/run --prompt "ROMEO:"
```

At the `tiny` preset the output is mostly gibberish with the right shape: line breaks, capitalized names and colons. Then run `python train.py` (the `small` preset, a few minutes on an Apple Silicon or NVIDIA GPU) and the text starts to look like Shakespeare.

## What's in here

| File | What it does |
|---|---|
| `tinylm/tokenizer.py` | Character-level and byte-pair-encoding (BPE) tokenizers, pure Python |
| `tinylm/model.py` | The transformer: causal self-attention, MLP, blocks, and the GPT wrapper with `generate()` |
| `tinylm/data.py` | Train/val split and random batches (each target is the next token) |
| `train.py` | Training loop: AdamW, warmup and cosine learning-rate decay, gradient clipping, evaluation, checkpoints |
| `sample.py` | Generate text from a checkpoint, with temperature and top-k |
| `scripts/plot_loss.py` | Plot train and val loss from a run (`pip install matplotlib`) |
| `tests/` | Checks that the tokenizers round-trip, attention is causal, and the model can memorize a batch |

Presets: `tiny` (~0.1M parameters, CPU), `small` (~0.8M, laptop GPU), `base` (~10M, use a real GPU such as a free Colab T4). Any setting can be overridden, for example `python train.py --n-layer 6 --lr 5e-4`.

## Roadmap

Each milestone ends with something you can show: a loss curve, generated samples, or a short write-up in this README.

**1. Get it running and read every line.**
Train `tiny` and `small`. Read `model.py` top to bottom until you can explain why the attention mask is lower-triangular and why the targets are the inputs shifted by one. Plot the loss. *Done when:* val loss drops below ~2.0 on Shakespeare (char-level, `small`) and you can explain each line of `CausalSelfAttention`.

**2. Break it on purpose.**
Run one experiment at a time and record the val loss: remove the causal mask (training loss collapses because the model can see the answer; why is generation then useless?), remove positional embeddings, remove the residual connections, drop the learning-rate warmup, crank dropout to 0.5. *Done when:* a results table in this README shows what each component contributes.

**3. Switch to BPE.**
Train with `--tokenizer bpe --vocab-size 1024`. Compare tokens per character, training speed and sample quality against char-level. Note that loss per token isn't comparable across tokenizers. Work out how to compare them fairly (hint: bits per character). *Done when:* you have a fair comparison.

**4. Scale up on a GPU.**
Move to Google Colab (free T4) and train `base`. Swap the hand-written attention for `F.scaled_dot_product_attention` and measure the speedup. Add mixed precision (`torch.autocast`). *Done when:* you have a ~10M-parameter model and timing numbers for each optimization.

**5. Train on your own data.**
Pick a corpus you care about, such as quantum-computing arXiv abstracts, Qiskit source code, or your own notes. Clean it, train a BPE tokenizer on it, and train a model. *Done when:* it generates plausible text in that domain and you've written up what the data cleaning involved, which usually turns out to be the hard part.

**6. Evaluate and fine-tune.**
Hold out a test set and report perplexity. Then fine-tune: start from your Shakespeare checkpoint and continue training on a small new corpus, comparing against training from scratch. *Done when:* you can show when fine-tuning helps and when it doesn't.

**Stretch goals:** key-value caching for faster generation, rotary position embeddings, a tiny instruction-tuning dataset you write yourself, or loading GPT-2 weights into this architecture to check your implementation matches.

## Good references

- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube) and his nanoGPT repo, the closest companion to this project
- Vaswani et al., *Attention Is All You Need* (2017), the original transformer paper
- Jay Alammar, *The Illustrated Transformer*, for intuition about attention
