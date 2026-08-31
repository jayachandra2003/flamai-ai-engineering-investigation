# Part A1: Multilingual Evaluation Corpus

## 1. Dataset Selection & Overview

To conduct a controlled, reproducible, and linguistically fair evaluation of tokenizer efficiency across multiple languages, we selected the **FLORES-200 Evaluation Benchmark** (published by Meta AI / NLLB Team).

### Why FLORES-200?
1. **Parallel & Content-Controlled:** FLORES-200 consists of multi-way parallel sentences professionally translated by human linguists. Measuring tokenization on parallel sentences ensures that differences in token count, character count, or fertility reflect genuine linguistic and tokenization properties rather than topical, domain, or semantic divergence.
2. **Standardized Typological Coverage:** It includes high-quality native script translations for English (`eng_Latn`), Indo-Aryan Hindi (`hin_Deva`), and Dravidian languages Kannada (`kan_Knda`) and Telugu (`tel_Telu`).
3. **Open & Reproducible:** The dataset is publicly hosted by Meta and packaged directly as raw parallel text files.

---

## 2. Configuration & Provenance

* **Dataset Name:** FLORES-200 (No Language Left Behind / NLLB Evaluation Benchmark)
* **Official URL:** `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`
* **Paper / Reference:** NLLB Team et al., *No Language Left Behind: Scaling Human-Centered Machine Translation* (2022).
* **Archive Files Verified:** `metadata_dev.tsv` (997 rows) and `metadata_devtest.tsv` (1012 rows).
* **Split Selected:** `dev` split (997 parallel sentences across all 204 languages).
* **Languages & BCP-47 / NLLB Codes:**
  * **English (`eng`):** `eng_Latn` (Latin script)
  * **Hindi (`hin`):** `hin_Deva` (Devanagari script, Indo-Aryan family)
  * **Kannada (`kan`):** `kan_Knda` (Kannada script, Dravidian family)
  * **Telugu (`tel`):** `tel_Telu` (Telugu script, Dravidian family)
* **Domain / Sources in `dev`:** 281 distinct source articles verified from `metadata_dev.tsv`, sourced from `wikinews`, `wikibooks`, and `wikivoyage` covering news, science, health, travel, history, and culture. (Across all splits including hidden test, FLORES-200 references 842 articles).
* **Text Encoding:** UTF-8 throughout.

---

## 3. Preprocessing & Alignment Pipeline

The corpus was prepared using [`scripts/prepare_corpus.py`](partA/scripts/prepare_corpus.py) through the following pipeline:
1. **In-Memory HTTP Fetch:** The official 25.58 MB archive (`flores200_dataset.tar.gz`) is downloaded in full via HTTP GET into memory (`io.BytesIO`).
2. **Extraction:** The 4 target `.dev` language files are extracted.
3. **Unicode Normalization:** All text is canonicalized using standard Unicode NFC (`unicodedata.normalize('NFC', text)`).
4. **Sanitization:** Leading/trailing whitespaces are stripped, empty lines are excluded, and each sentence is written on a single line with `\n` line endings.
5. **Preservation:** The original line ordering from the official FLORES-200 archive is strictly preserved.

### Alignment Distinction:
- **What the dataset guarantees:** Per the official FLORES-200 documentation, the sentences in `dev/<lang>.dev` are ordered identically across all languages, representing professional human translations of the exact same underlying source sentence.
- **What our script verifies:** Our preparation script strictly verifies that all 4 generated files contain exactly 997 non-empty lines, have zero encoding errors, are NFC-normalized, and preserve the exact 1-to-1 sequential line order of the official archive member files.

---

## 4. Corpus Statistics

Statistics directly recalculated from the generated files in [`partA/corpus/`](partA/corpus/):

| Language | ISO / Script | Family | Sentences (Lines) | Whitespace Words (`split()`) | Unicode Chars | Text Bytes (UTF-8) | Total File Size (Bytes) | Words / Sent | Chars / Sent | Bytes / Char |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **English** | `eng_Latn` | Germanic (Indo-European) | 997 | 20,954 | 125,194 | 125,290 | 126,287 | 21.02 | 125.57 | 1.00 |
| **Hindi** | `hin_Deva` | Indo-Aryan (Indo-European) | 997 | 24,607 | 125,495 | 322,640 | 323,637 | 24.68 | 125.87 | 2.57 |
| **Kannada** | `kan_Knda` | South Dravidian | 997 | 15,430 | 131,749 | 357,408 | 358,405 | 15.48 | 132.15 | 2.71 |
| **Telugu** | `tel_Telu` | South-Central Dravidian | 997 | 16,388 | 127,172 | 338,804 | 339,801 | 16.44 | 127.55 | 2.66 |

*Note on "Words": "Whitespace Words" is computed via Python `str.split()`. For agglutinative Dravidian languages (Kannada, Telugu), whitespace-separated tokens represent compound/inflected morphological units, which is why whitespace word counts are lower than English or Hindi for the same semantic content.*

---

## 5. What This Corpus Cannot Tell Us (Limitations & Caveats)

1. **Domain & Register Bias:** FLORES-200 sentences are drawn from formal, edited written text (encyclopedic, journalistic, and informational articles from Wikimedia platforms). They do not reflect informal colloquial speech, social media slang, conversational chat dialog, or domain-specific jargon (e.g., enterprise customer support, legal docs, or code-heavy prompts).
2. **Code-Mixing & Transliteration (Hinglish/Kanglish/Tanglish):** Real-world Indic LLM traffic heavily features code-switching (e.g., Hindi written in Latin script, or mixed Hindi-English sentences). FLORES-200 provides clean, native-script text and does not capture romanized Indic tokenization patterns.
3. **Prompt Framing & System Context:** Real production inference payloads contain structured templates (JSON keys, markdown tags, XML tags, system prompts). Parallel sentence corpora measure isolated natural language efficiency rather than holistic serving request composition.
4. **Sample Size Scope:** 997 parallel sentences provide substantially broader coverage than the initial smoke-test sample, while remaining a finite benchmark sample.
