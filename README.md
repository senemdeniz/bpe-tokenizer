# BPE Tokenizer + Mineral Adı Üretme

Bu klasör, ödevin iki parçasını birleştirir:

1. **Kendi metnimizle BPE tokenizer** — 4104 mineral adı üzerinde byte-level BPE
   (Byte-Pair Encoding) tokenizer eğitildi.
2. **Minyatür model eğitimi** — reponun `TinyQwen` (Qwen3 tarzı) modeli, char-seviyeli
   tokenizer yerine bu BPE tokenizer'ı kullanarak yeni mineral adları üretmeyi öğrendi.

Referans: [HuggingFace LLM Course — Bölüm 6/5 (BPE)](https://huggingface.co/learn/llm-course/en/chapter6/5)
ve reponun kökündeki `train_hf_tokenizer.py`.

## Neden BPE? (char-seviyeliden farkı)

Reponun varsayılanı **char-seviyeli**: her token tek bir harf. BPE ise verideki en sık
geçen harf çiftlerini adım adım birleştirerek **alt-kelime (subword)** token'ları öğrenir.
Minerallerde bu çok işe yarar; çünkü adların büyük kısmı `-ite`, `-lite`, `-ine` gibi
eklerle biter ve BPE bunları tek token yapar:

```
abelsonite -> ['ab', 'el', 'sonite']      (3 token)
pyrite     -> ['p', 'y', 'rite']          (3 token)
aegirine   -> ['a', 'e', 'g', 'ir', 'ine'] (5 token)
```

## Dosyalar

| Dosya | Ne yapar |
| ----- | -------- |
| `mineraller.txt` | Veri: 4104 mineral adı, küçük harf, satır başına bir ad |
| `train_bpe_tokenizer.py` | BPE tokenizer'ı eğitir → `mineral_bpe.json` + `mineral_bpe_hf/` |
| `bpe_tokenizer.py` | Tokenizer'ı reponun arayüzüne saran `BPETokenizer` sınıfı |
| `train.py` | `TinyQwen`'i BPE token'larıyla eğitir → `tiny_qwen_mineral.pt` |
| `generate.py` | Checkpoint'ten yeni mineral adları üretir |
| `qwen3_model/` | Model mimarisi (Qwen3 tarzı). [malibayram/single_letter_transformers](https://github.com/malibayram/single_letter_transformers) reposunun `qwen3/` klasöründen kopyalandı |

> Bu klasör **self-contained**'dır: `qwen3_model/` sayesinde çalışmak için başka
> bir repoya ihtiyaç duymaz. Tek başına GitHub'a yüklenebilir.

## Çalıştırma

```bash
# 1) BPE tokenizer'ı eğit (sözlük boyutu 512)
python3 train_bpe_tokenizer.py

# 2) Modeli eğit (~3000 adım, CPU'da birkaç saniye)
python3 train.py

# 3) Yeni mineral adları üret
python3 generate.py 20        # 20 ad
python3 generate.py 30 0.6    # düşük temperature = daha "güvenli" adlar
```

> Not: torch + tokenizers + transformers'ın hepsinin bulunduğu bir Python gerekir.
> Bu makinede `/usr/bin/python3` (3.9) uygundu.

## Sonuçlar (örnek çalışma, CPU)

- Kayıp (loss): tekdüze taban `ln 512 ≈ 6.24` → **~2.48**
- Üretilenlerin **%97'si** eğitim verisinde olmayan **yeni** adlar (ezber değil)
- Üretilenlerin **%96'sı** gerçek mineral geleneğine uygun `-ite` ile bitiyor

Örnek üretimler:

```
molybdite, kogatovite, krinite, khaniite, habsite,
jasplaite, kallionite, nickelinikite, spertsite, felbenbachite
```

## Nasıl çalışıyor? (kısa akış)

```
mineraller.txt
   │  train_bpe_tokenizer.py  (byte-level BPE, vocab=512, özel token: <bos> <eos> ...)
   ▼
mineral_bpe.json  ──►  BPETokenizer.encode("pyrite") = [p, y, rite]
   │
   │  train.py: korpus = her adın ardına <eos>  →  TinyQwen (Qwen3: RMSNorm, GQA, RoPE, SwiGLU)
   ▼
tiny_qwen_mineral.pt  ──►  generate.py: <eos>'tan başla, <eos>'ta dur  →  yeni ad
```
