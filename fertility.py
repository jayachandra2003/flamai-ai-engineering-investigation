#!/usr/bin/env python3
"""
fertility.py -- tokenizer fertility benchmark (v0)

Computes tokenizer fertility (tokens per word) and compression
(tokens per character) for one or more language corpora.

Usage:
    python fertility.py --corpus eng=corpus_sample/eng_sample.txt \
                        --corpus hin=corpus_sample/hin_sample.txt \
                        --tokenizer gpt2

Tokenizers:
    gpt2            -> tiktoken "gpt2" encoding (default)
    hf:<repo_id>    -> any HuggingFace tokenizer, e.g. hf:xlm-roberta-base

Author: previous intern (v0, "good enough for the deck")
"""

import argparse
import random
import sys
import unicodedata

random.seed(1337)  # reproducibility


def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken

        enc = tiktoken.get_encoding(spec)
        return enc.encode


def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # normalize just in case -- some corpora are messy
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def analyze(lines, encode):
    """Return (fertility, tokens_per_char) averaged over lines."""
    per_line_fertility = []
    per_line_tpc = []
    for line in lines:
        # lowercase so casing doesn't add noise to the comparison
        line = line.lower()
        tokens = encode(line)
        words = line.split(" ")
        chars = len(line)
        per_line_fertility.append(len(tokens) / len(words))
        per_line_tpc.append(len(tokens) / chars)
    n = len(per_line_fertility)
    return sum(per_line_fertility) / n, sum(per_line_tpc) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and path, e.g. eng=data/eng.txt (repeatable)",
    )
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    encode = load_tokenizer(args.tokenizer)

    print(f"tokenizer: {args.tokenizer}")
    print(f"{'lang':<8}{'fertility (tok/word)':>22}{'tok/char':>12}")
    print("-" * 42)
    results = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path)
        fert, tpc = analyze(lines, encode)
        results[lang] = (fert, tpc)
        print(f"{lang:<8}{fert:>22.2f}{tpc:>12.3f}")

    if len(results) >= 2:
        langs = list(results)
        base = langs[0]
        print()
        for lang in langs[1:]:
            ratio = results[lang][0] / results[base][0]
            print(f"{lang} is {ratio:.2f}x the fertility of {base} "
                  f"({'worse' if ratio > 1 else 'better'} tokenization)")


if __name__ == "__main__":
    main()
