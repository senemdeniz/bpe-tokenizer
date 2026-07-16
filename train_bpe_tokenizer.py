"""Mineral adlarindan bir BPE (Byte-Pair Encoding) tokenizer egitir.

Bu, HuggingFace LLM kursu Bolum 6/5 ("WordPiece / BPE") ve reponun kokundeki
`train_hf_tokenizer.py` ornegiyle ayni yaklasimdir: byte-level BPE.

BPE nasil calisir (ozet):
  1. Her metin once tek tek byte'lara (byte-level alfabe) bolunur.
  2. En sik gecen bitisik token cifti bulunup TEK bir yeni token'a birlestirilir.
  3. Bu islem, sozluk (vocab) istenen boyuta ulasana kadar tekrarlanir.
Sonucta "ite", "lite", "ine" gibi minerallerde cok gecen ekler tek token olur.

Calistir:
    python3 train_bpe_tokenizer.py

Ciktilar:
    mineral_bpe.json          -> ham tokenizers backend (BPETokenizer bunu okur)
    mineral_bpe_hf/           -> AutoTokenizer.from_pretrained ile yuklenebilen surum
"""

import os

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

HERE = os.path.dirname(__file__)
TEXT_FILE = os.path.join(HERE, "mineraller.txt")
VOCAB_SIZE = 512          # kucuk ama "ite/lite/ine" gibi ekleri yakalamaya yeter
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

# ---------------------------------------------------------------------------
# 1. Byte-level BPE tokenizer'i olustur ve egit.
# ---------------------------------------------------------------------------
backend = Tokenizer(BPE(unk_token="<unk>"))
backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
backend.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=ByteLevel.alphabet(),   # 256 byte tabanini garanti eder
    show_progress=True,
)
backend.train([TEXT_FILE], trainer)

# ---------------------------------------------------------------------------
# 2. Transformers/AutoTokenizer'in anlayacagi bir sarmalayiciya koy.
# ---------------------------------------------------------------------------
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
)

# ---------------------------------------------------------------------------
# 3. Kaydet (hem ham json hem de HF klasoru).
# ---------------------------------------------------------------------------
json_path = os.path.join(HERE, "mineral_bpe.json")
backend.save(json_path)
tokenizer.save_pretrained(os.path.join(HERE, "mineral_bpe_hf"))
print(f"\nsozluk boyutu: {backend.get_vocab_size()}")
print(f"kaydedildi: {json_path}")
print(f"kaydedildi: {os.path.join(HERE, 'mineral_bpe_hf')}")

# ---------------------------------------------------------------------------
# 4. Hizli kontrol: birkac mineral adini token'lara bol.
# ---------------------------------------------------------------------------
print("\n--- ornek tokenizasyon ---")
for word in ["abelsonite", "malachite", "pyrite", "aegirine"]:
    enc = backend.encode(word)
    print(f"{word:14s} -> {enc.tokens}  ({len(enc.ids)} token)")
