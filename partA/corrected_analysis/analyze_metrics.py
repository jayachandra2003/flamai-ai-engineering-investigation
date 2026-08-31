#!/usr/bin/env python3
"""
analyze_metrics.py -- Comprehensive Metric & Tokenizer Evaluation for Part A3.

Evaluates candidate metrics across:
  - 4 languages: English (eng), Hindi (hin), Kannada (kan), Telugu (tel)
  - 997 parallel FLORES-200 sentences (partA/corpus/)
  - 4 representative tokenizers:
      1. GPT-2 (50,257 vocab, baseline)
      2. Meta-Llama-3-8B (128,000 vocab, representative 128k BPE matching FLM-4B vocab scale)
      3. Qwen2.5-7B (151,643 vocab, multilingual BPE)
      4. XLM-RoBERTa-base (250,002 vocab, multilingual SentencePiece)
"""

import os
import sys
import json
import unicodedata
import tiktoken
from transformers import AutoTokenizer

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_DIR = os.path.join(BASE_DIR, "partA", "corpus")
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

LANGS = ["eng", "hin", "kan", "tel"]
CORPUS_FILES = {lang: os.path.join(CORPUS_DIR, f"{lang}.txt") for lang in LANGS}


def load_corpus():
    corpus = {}
    for lang, path in CORPUS_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            lines = [unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()]
        corpus[lang] = lines
    return corpus


def get_tokenizers():
    print("Loading tokenizers for Part A3 analysis...")
    return {
        "gpt2": {
            "name": "GPT-2 (tiktoken)",
            "vocab_size": 50257,
            "family": "Byte-level BPE",
            "encode": tiktoken.get_encoding("gpt2").encode
        },
        "llama3_128k": {
            "name": "Meta-Llama-3-8B (HF)",
            "vocab_size": 128000,
            "family": "Tiktoken / Byte-level BPE (128k scale)",
            "encode": lambda s, tok=AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B"): tok.encode(s, add_special_tokens=False)
        },
        "qwen2.5_152k": {
            "name": "Qwen2.5-7B (HF)",
            "vocab_size": 151643,
            "family": "Byte-level BPE (152k multilingual)",
            "encode": lambda s, tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"): tok.encode(s, add_special_tokens=False)
        },
        "xlm_roberta_250k": {
            "name": "XLM-RoBERTa-base (HF)",
            "vocab_size": 250002,
            "family": "Unigram SentencePiece (250k multilingual)",
            "encode": lambda s, tok=AutoTokenizer.from_pretrained("xlm-roberta-base"): tok.encode(s, add_special_tokens=False)
        }
    }


def analyze_all(corpus, tokenizers):
    full_results = {
        "metadata": {
            "corpus_name": "FLORES-200 dev split (clean NFC)",
            "sentence_count": len(corpus["eng"]),
            "languages": LANGS
        },
        "corpus_stats": {},
        "tokenizer_metrics": {},
        "cross_language_comparison": {}
    }

    # 1. Corpus ground truth statistics
    for lang, lines in corpus.items():
        words_split = sum(len(l.split()) for l in lines)
        chars = sum(len(l) for l in lines)
        graphemes = sum(sum(1 for c in l if unicodedata.category(c) not in ('Mn', 'Me')) for l in lines)
        bytes_utf8 = sum(len(l.encode("utf-8")) for l in lines)
        full_results["corpus_stats"][lang] = {
            "lines": len(lines),
            "whitespace_words": words_split,
            "unicode_chars": chars,
            "grapheme_clusters": graphemes,
            "utf8_bytes": bytes_utf8,
            "words_per_sentence": round(words_split / len(lines), 2),
            "chars_per_sentence": round(chars / len(lines), 2),
            "graphemes_per_sentence": round(graphemes / len(lines), 2),
            "bytes_per_char": round(bytes_utf8 / chars, 2)
        }

    # 2. Tokenizer evaluations
    for tok_id, tok_meta in tokenizers.items():
        enc = tok_meta["encode"]
        tok_data = {
            "tokenizer_name": tok_meta["name"],
            "vocab_size": tok_meta["vocab_size"],
            "family": tok_meta["family"],
            "languages": {}
        }

        # First pass: collect per-language metrics
        for lang, lines in corpus.items():
            line_token_counts = []
            line_word_counts = []
            line_char_counts = []
            line_grapheme_counts = []
            line_byte_counts = []
            
            per_line_fert = []
            per_line_tpc = []
            per_line_tpg = []
            per_line_tpb = []

            for l in lines:
                toks = enc(l)
                n_toks = len(toks)
                w = len(l.split())
                c = len(l)
                g = sum(1 for ch in l if unicodedata.category(ch) not in ('Mn', 'Me'))
                b = len(l.encode("utf-8"))

                line_token_counts.append(n_toks)
                line_word_counts.append(w)
                line_char_counts.append(c)
                line_grapheme_counts.append(g)
                line_byte_counts.append(b)

                per_line_fert.append(n_toks / w)
                per_line_tpc.append(n_toks / c)
                per_line_tpg.append(n_toks / g)
                per_line_tpb.append(n_toks / b)

            tot_tokens = sum(line_token_counts)
            tot_words = sum(line_word_counts)
            tot_chars = sum(line_char_counts)
            tot_graphemes = sum(line_grapheme_counts)
            tot_bytes = sum(line_byte_counts)

            macro_fert = sum(per_line_fert) / len(per_line_fert)
            micro_fert = tot_tokens / tot_words
            macro_tpc = sum(per_line_tpc) / len(per_line_tpc)
            micro_tpc = tot_tokens / tot_chars
            macro_tpg = sum(per_line_tpg) / len(per_line_tpg)
            micro_tpg = tot_tokens / tot_graphemes
            macro_tpb = sum(per_line_tpb) / len(per_line_tpb)
            micro_tpb = tot_tokens / tot_bytes

            tok_data["languages"][lang] = {
                "total_tokens": tot_tokens,
                "macro_fertility_tok_per_word": round(macro_fert, 4),
                "micro_fertility_tok_per_word": round(micro_fert, 4),
                "macro_tok_per_char": round(macro_tpc, 4),
                "micro_tok_per_char": round(micro_tpc, 4),
                "macro_tok_per_grapheme": round(macro_tpg, 4),
                "micro_tok_per_grapheme": round(micro_tpg, 4),
                "macro_tok_per_byte": round(macro_tpb, 4),
                "micro_tok_per_byte": round(micro_tpb, 4),
            }

        # Second pass: compute relative ratios vs English
        eng_tokens = tok_data["languages"]["eng"]["total_tokens"]
        eng_micro_fert = tok_data["languages"]["eng"]["micro_fertility_tok_per_word"]
        eng_micro_tpc = tok_data["languages"]["eng"]["micro_tok_per_char"]
        eng_micro_tpg = tok_data["languages"]["eng"]["micro_tok_per_grapheme"]
        eng_micro_tpb = tok_data["languages"]["eng"]["micro_tok_per_byte"]

        for lang, l_metrics in tok_data["languages"].items():
            l_metrics["token_expansion_ratio_vs_eng"] = round(l_metrics["total_tokens"] / eng_tokens, 3)
            l_metrics["fertility_ratio_vs_eng"] = round(l_metrics["micro_fertility_tok_per_word"] / eng_micro_fert, 3)
            l_metrics["tok_per_char_ratio_vs_eng"] = round(l_metrics["micro_tok_per_char"] / eng_micro_tpc, 3)
            l_metrics["tok_per_grapheme_ratio_vs_eng"] = round(l_metrics["micro_tok_per_grapheme"] / eng_micro_tpg, 3)
            l_metrics["tok_per_byte_ratio_vs_eng"] = round(l_metrics["micro_tok_per_byte"] / eng_micro_tpb, 3)

        full_results["tokenizer_metrics"][tok_id] = tok_data

    return full_results


def print_summary_tables(results):
    print("\n" + "=" * 115)
    print("PART A3: CORRECTED MULTILINGUAL METRIC ANALYSIS (FLORES-200, 997 Parallel Sentences)")
    print("=" * 115)

    for tok_id, t_data in results["tokenizer_metrics"].items():
        print(f"\n>>> Tokenizer: {t_data['tokenizer_name']} | Vocab: {t_data['vocab_size']:,} | Family: {t_data['family']}")
        header = f"{'Lang':<6}{'Total Toks':>12}{'Tok/Word':>10}{'Tok/Char':>10}{'Tok/Graph':>10}{'Tok/Byte':>10}{'Tok Expansion vs Eng':>22}{'Fertility vs Eng':>18}"
        print(header)
        print("-" * len(header))
        for lang in LANGS:
            m = t_data["languages"][lang]
            print(f"{lang:<6}{m['total_tokens']:>12d}{m['micro_fertility_tok_per_word']:>10.2f}{m['micro_tok_per_char']:>10.3f}{m['micro_tok_per_grapheme']:>10.3f}{m['micro_tok_per_byte']:>10.3f}{m['token_expansion_ratio_vs_eng']:>21.2f}x{m['fertility_ratio_vs_eng']:>17.2f}x")
        print("-" * len(header))

    print("\n" + "=" * 105)
    print("DENOMINATOR COMPARISON: Fertility Ratio vs Total Token Expansion Ratio")
    print("=" * 105)
    header_d = f"{'Tokenizer':<24}{'Lang':<6}{'Total Tokens':>14}{'Token Ratio vs Eng':>20}{'Fertility (Tok/Word)':>22}{'Fert Ratio vs Eng':>20}{'Divergence (%)':>16}"
    print(header_d)
    print("-" * len(header_d))
    for tok_id, t_data in results["tokenizer_metrics"].items():
        for lang in ["hin", "kan", "tel"]:
            m = t_data["languages"][lang]
            tok_ratio = m["token_expansion_ratio_vs_eng"]
            fert_ratio = m["fertility_ratio_vs_eng"]
            div_pct = ((fert_ratio - tok_ratio) / tok_ratio) * 100
            print(f"{t_data['tokenizer_name']:<24}{lang:<6}{m['total_tokens']:>14d}{tok_ratio:>19.2f}x{m['micro_fertility_tok_per_word']:>21.2f}{fert_ratio:>19.2f}x{div_pct:>15.1f}%")
        print("-" * len(header_d))


def main():
    corpus = load_corpus()
    tokenizers = get_tokenizers()
    results = analyze_all(corpus, tokenizers)
    
    out_file = os.path.join(RESULTS_DIR, "corrected_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete results to: {out_file}")

    print_summary_tables(results)


if __name__ == "__main__":
    main()
