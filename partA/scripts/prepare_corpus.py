#!/usr/bin/env python3
"""
prepare_corpus.py -- Downloads, preprocesses, and validates the multilingual
parallel evaluation corpus for Part A1 of the FlamAI Audit.

Languages:
  - eng: English (Latin script, eng_Latn)
  - hin: Hindi (Devanagari script, hin_Deva)
  - kan: Kannada (Kannada script, kan_Knda)
  - tel: Telugu (Telugu script, tel_Telu)

Source:
  FLORES-200 Benchmark (Meta AI / NLLB Team)
  URL: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
"""

import io
import os
import sys
import tarfile
import unicodedata
import requests

# Ensure UTF-8 output on Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

DATASET_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

LANG_MAP = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "tel": "tel_Telu",
}

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "corpus"))


def download_and_extract_corpus():
    print(f"Fetching FLORES-200 dataset into memory from {DATASET_URL}...")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    print(f"Downloaded {len(response.content):,} bytes.")

    tar = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extracted_corpus = {}
    for short_lang, flores_code in LANG_MAP.items():
        member_path = f"./flores200_dataset/dev/{flores_code}.dev"
        file_obj = tar.extractfile(member_path)
        if file_obj is None:
            raise RuntimeError(f"Could not find member {member_path} in FLORES archive")

        lines = []
        for raw_line in file_obj:
            text = raw_line.decode("utf-8").strip()
            if text:
                # Unicode NFC normalization
                text = unicodedata.normalize("NFC", text)
                lines.append(text)

        extracted_corpus[short_lang] = lines

    # Verify line counts and 1-to-1 alignment
    line_counts = {lang: len(lines) for lang, lines in extracted_corpus.items()}
    print(f"Line counts extracted: {line_counts}")
    if len(set(line_counts.values())) != 1:
        raise ValueError(f"Line count mismatch across languages: {line_counts}")

    # Write cleaned, aligned files
    for short_lang, lines in extracted_corpus.items():
        out_path = os.path.join(OUTPUT_DIR, f"{short_lang}.txt")
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"Saved: {out_path} ({len(lines)} lines)")

    return extracted_corpus


def analyze_corpus(corpus):
    print("\n" + "=" * 94)
    print("MULTILINGUAL EVALUATION CORPUS STATISTICS (FLORES-200 dev)")
    print("=" * 94)
    header = f"{'Lang':<6}{'Lines':>8}{'Whitespace Words':>18}{'Chars':>10}{'Text Bytes':>14}{'Words/Sent':>12}{'Chars/Sent':>12}{'Bytes/Char':>12}"
    print(header)
    print("-" * len(header))

    for lang, lines in corpus.items():
        num_lines = len(lines)
        total_words = sum(len(line.split()) for line in lines)
        total_chars = sum(len(line) for line in lines)
        total_bytes = sum(len(line.encode("utf-8")) for line in lines)

        words_per_sent = total_words / num_lines if num_lines else 0
        chars_per_sent = total_chars / num_lines if num_lines else 0
        bytes_per_char = total_bytes / total_chars if total_chars else 0

        print(f"{lang:<6}{num_lines:>8d}{total_words:>18d}{total_chars:>10d}{total_bytes:>14d}{words_per_sent:>12.2f}{chars_per_sent:>12.2f}{bytes_per_char:>12.2f}")

    print("-" * len(header))
    print("Alignment Note: All 4 files preserve the exact sentence order of FLORES-200 dev.")
    print("Word Count Note: Words are computed via Python whitespace splitting (str.split()).")
    print("=" * 94)


def main():
    corpus = download_and_extract_corpus()
    analyze_corpus(corpus)


if __name__ == "__main__":
    main()
