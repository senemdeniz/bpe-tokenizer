"""TinyQwen modelini BPE token'lariyla mineral adi uretmesi icin egitir.

Reponun qwen3/train.py'siyle ayni iskelet; TEK fark tokenizer:
char-seviyeli yerine kendi egittigimiz BPE tokenizer'i kullaniyoruz.

Once tokenizer'i egit:   python3 train_bpe_tokenizer.py
Sonra modeli egit:       python3 train.py
Uret:                    python3 generate.py 20
"""

import os
import sys

import torch

# TinyQwen ve arkadaslarini yerel qwen3_model/ klasorunden al (self-contained:
# bu dosyalar single_letter_transformers/qwen3'ten kopyalandi, disari bagimlilik yok).
BPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BPE_DIR, "qwen3_model"))

from config import ModelConfig            # noqa: E402
from model import TinyQwen                 # noqa: E402
from bpe_tokenizer import BPETokenizer     # noqa: E402

# ---------------------------------------------------------------------------
# Hiperparametreler
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "mineraller.txt")
BATCH_SIZE = 64
BLOCK_SIZE = 24        # baglam uzunlugu (<= cfg.max_seq_len)
STEPS = 3000
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Tokenizer (BPE). bpe_tokenizer.py + mineral_bpe.json.
# ---------------------------------------------------------------------------
tokenizer = BPETokenizer()
vocab_size = tokenizer.vocab_size

# Korpusu tek bir token akisina cevir: her ismin ardina <eos> koy.
# Boylece model bir ismin nerede bittigini ve yeni bir ismin nerede
# basladigini ogrenir (char surumundeki "\n" ile ayni is).
names = [line.strip() for line in open(DATA_FILE, encoding="utf-8") if line.strip()]
stream = []
for name in names:
    stream.extend(tokenizer.encode(name))
    stream.append(tokenizer.eos_id)
data = torch.tensor(stream, dtype=torch.long)
print(f"toplam isim: {len(names)}   toplam token: {len(data)}   "
      f"isim basi ort. token: {len(data)/len(names):.1f}")


def get_batch():
    """BATCH_SIZE rastgele pencere. Hedef = girdinin bir saga kaydirilmisi."""
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=32)
model = TinyQwen(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device={device}  vocab_size={vocab_size}  parametre={n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------------
# Ornekleme: <eos>'tan baslayip yeni mineral adlari uret.
# ---------------------------------------------------------------------------
def sample_names(n: int = 10, max_new_tokens: int = 32):
    model.eval()
    start = torch.full((n, 1), tokenizer.eos_id, dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=max_new_tokens, temperature=1.0,
                         top_k=None, eos_id=tokenizer.eos_id)
    model.train()
    names_out = []
    for row in out.tolist():
        ids = row[1:]                       # bastaki <eos>'u at
        if tokenizer.eos_id in ids:         # ilk <eos>'ta kes
            ids = ids[:ids.index(tokenizer.eos_id)]
        names_out.append(tokenizer.decode(ids))
    return names_out


# ---------------------------------------------------------------------------
# Egitim dongusu
# ---------------------------------------------------------------------------
for step in range(1, STEPS + 1):
    x, y = get_batch()
    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == 1:
        print(f"step {step:5d}  loss {loss.item():.4f}")

baseline = torch.log(torch.tensor(float(vocab_size)))
print(f"\ntekduze tahmin taban kaybi (ln {vocab_size}): {baseline:.4f}")
print("\nuretilen mineral adlari:")
for name in sample_names(15):
    print("  ", name)

ckpt_path = os.path.join(HERE, "tiny_qwen_mineral.pt")
torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
print(f"\ncheckpoint kaydedildi: {ckpt_path}")
