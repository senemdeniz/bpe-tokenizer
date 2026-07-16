"""Egitilmis BPE tokenizer'i reponun bekledigi arayuze saran ince bir sinif.

Reponun char-seviyeli `CharTokenizer`'i gibi `encode/decode/vocab_size/eos_id`
sunar; boylece TinyQwen modeli hicbir degisiklik olmadan calisir. Tek fark:
token'lar tek harf degil, BPE ile ogrenilen alt-kelimelerdir (ornegin "ite").

`<eos>` token'i hem isim ayiraci hem de dur (stop) isareti olarak kullanilir --
char-seviyeli surumde "\n"in yaptigi isi yapar.
"""

import os

from tokenizers import Tokenizer

HERE = os.path.dirname(__file__)
DEFAULT_JSON = os.path.join(HERE, "mineral_bpe.json")


class BPETokenizer:
    def __init__(self, json_path: str = DEFAULT_JSON):
        self.backend = Tokenizer.from_file(json_path)
        self.vocab_size = self.backend.get_vocab_size()
        self.bos_id = self.backend.token_to_id("<bos>")
        self.eos_id = self.backend.token_to_id("<eos>")

    def encode(self, s: str) -> list[int]:
        """Metni BPE token id dizisine cevirir (ozel token EKLEMEZ)."""
        return self.backend.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        """Token id'lerini metne geri cevirir; ozel token'lari atar."""
        return self.backend.decode(ids, skip_special_tokens=True)

    def encode_name(self, name: str) -> list[int]:
        """Tek bir mineral adini <bos> ... <eos> ile sarar."""
        return [self.bos_id] + self.encode(name) + [self.eos_id]
