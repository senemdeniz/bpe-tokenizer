# BPE Tokenizer + Mineral Name Generator

A small end-to-end language-modelling project in two parts:

1. **A BPE tokenizer trained on my own corpus**: a byte-level BPE (Byte-Pair Encoding)
   tokenizer trained on 4,104 mineral names.
2. **Training a tiny language model**: a Qwen3-style transformer (`TinyQwen`) that uses
   this BPE tokenizer instead of a character-level one and learns to generate new,
   realistic mineral names.

Reference: [Hugging Face LLM Course, Chapter 6.5 (BPE)](https://huggingface.co/learn/llm-course/en/chapter6/5).

## Why BPE? (vs. character-level)

A character-level tokenizer treats every letter as a token. BPE instead learns
**subword** tokens by repeatedly merging the most frequent pairs of symbols in the data.
This works well for minerals, because most names end in suffixes such as `-ite`,
`-lite` or `-ine`, which BPE learns as single tokens:

```
abelsonite -> ['ab', 'el', 'sonite']       (3 tokens)
pyrite     -> ['p', 'y', 'rite']           (3 tokens)
aegirine   -> ['a', 'e', 'g', 'ir', 'ine'] (5 tokens)
```

## Files

| File | What it does |
| ---- | ------------ |
| `mineraller.txt` | Data: 4,104 mineral names, lowercase, one per line |
| `train_bpe_tokenizer.py` | Trains the BPE tokenizer → `mineral_bpe.json` + `mineral_bpe_hf/` |
| `bpe_tokenizer.py` | `BPETokenizer` class that wraps the trained tokenizer for the model |
| `train.py` | Trains `TinyQwen` on the BPE tokens → `tiny_qwen_mineral.pt` |
| `generate.py` | Generates new mineral names from the checkpoint |
| `qwen3_model/` | Model architecture (Qwen3-style), adapted from the `qwen3/` folder of [malibayram/single_letter_transformers](https://github.com/malibayram/single_letter_transformers) |

The repository is self-contained: thanks to `qwen3_model/`, it does not depend on any
other repository.

## Getting started

Requirements: Python 3.9+ with `torch`, `tokenizers` and `transformers`.

```bash
pip install torch tokenizers transformers

# 1) Train the BPE tokenizer (vocabulary size 512)
python3 train_bpe_tokenizer.py

# 2) Train the model (~3,000 steps, a few seconds on CPU)
python3 train.py

# 3) Generate new mineral names
python3 generate.py 20        # 20 names
python3 generate.py 30 0.6    # lower temperature = "safer" names
```

## Results (sample run, CPU)

- Loss: from the uniform baseline `ln 512 ≈ 6.24` down to **~2.48**
- **97%** of generated names are **new**, i.e. not in the training data (not memorised)
- **96%** of generated names end in `-ite`, following real mineral naming conventions

Sample outputs:

```
molybdite, kogatovite, krinite, khaniite, habsite,
jasplaite, kallionite, nickelinikite, spertsite, felbenbachite
```

## How it works

```
mineraller.txt
   │  train_bpe_tokenizer.py  (byte-level BPE, vocab = 512, special tokens: <bos> <eos> ...)
   ▼
mineral_bpe.json  ──►  BPETokenizer.encode("pyrite") = [p, y, rite]
   │
   │  train.py: corpus = every name followed by <eos>  →  TinyQwen (Qwen3: RMSNorm, GQA, RoPE, SwiGLU)
   ▼
tiny_qwen_mineral.pt  ──►  generate.py: start from <eos>, stop at <eos>  →  new name
```

Each name is followed by an `<eos>` token, so the model learns where one name ends and
the next begins. Generation starts from `<eos>` and stops at the next `<eos>`.
