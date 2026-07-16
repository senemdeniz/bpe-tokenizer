"""Egitilmis checkpoint'ten yeni mineral adlari uretir.

Kullanim:
    python3 generate.py            # 20 isim, temperature 0.8
    python3 generate.py 30         # 30 isim
    python3 generate.py 30 0.6     # 30 isim, dusuk temperature (daha "guvenli")
"""

import os
import sys

import torch

BPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BPE_DIR, "qwen3_model"))

from model import TinyQwen                 # noqa: E402
from bpe_tokenizer import BPETokenizer     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

tokenizer = BPETokenizer()
ckpt = torch.load(os.path.join(HERE, "tiny_qwen_mineral.pt"),
                  map_location="cpu", weights_only=False)
model = TinyQwen(ckpt["cfg"])
model.load_state_dict(ckpt["model"])
model.eval()

start = torch.full((n, 1), tokenizer.eos_id, dtype=torch.long)
out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                     temperature=temperature, eos_id=tokenizer.eos_id)

for row in out.tolist():
    ids = row[1:]
    if tokenizer.eos_id in ids:
        ids = ids[:ids.index(tokenizer.eos_id)]
    print(tokenizer.decode(ids))
