#!/usr/bin/env python3
"""
run_audit.py -- Forensic Audit of fertility.py for Part A2 of the FlamAI Audit.

Performs controlled experiments across:
  1. Original starter corpora (corpus_sample/eng_sample.txt, corpus_sample/hin_sample.txt)
  2. Full parallel evaluation corpus (partA/corpus/eng.txt, hin.txt, kan.txt, tel.txt)

Evaluates:
  - Hypothesis A: Whitespace splitting (line.split(' ') vs line.split())
  - Hypothesis B: Lowercasing (line.lower() vs raw cased text)
  - Hypothesis C: Macro-average vs Micro-average aggregation
  - Hypothesis D: Denominator definitions (words vs Unicode chars vs UTF-8 bytes)
  - Hypothesis E: Tokenizer vocabulary scaling (GPT-2 vs LLaMA-3 128k vs Qwen2.5 150k vs XLM-R 250k)
  - Hypothesis F: Special tokens & prefix handling
  - Hypothesis G: Conceptual metric validity (Cross-linguistic comparison on parallel data)
  - Suspicious-but-correct: Unicode normalization (NFC vs NFD vs un-normalized)
"""

import os
import sys
import json
import csv
import unicodedata
import tiktoken
from transformers import AutoTokenizer

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Corpora paths
SAMPLE_CORPUS = {
    "eng": os.path.join(BASE_DIR, "corpus_sample", "eng_sample.txt"),
    "hin": os.path.join(BASE_DIR, "corpus_sample", "hin_sample.txt"),
}

FULL_CORPUS = {
    "eng": os.path.join(BASE_DIR, "partA", "corpus", "eng.txt"),
    "hin": os.path.join(BASE_DIR, "partA", "corpus", "hin.txt"),
    "kan": os.path.join(BASE_DIR, "partA", "corpus", "kan.txt"),
    "tel": os.path.join(BASE_DIR, "partA", "corpus", "tel.txt"),
}


def load_raw_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            l = raw.strip()
            if l:
                lines.append(l)
    return lines


# Load Tokenizers
print("Loading tokenizers...")
tokenizers = {
    "gpt2": {
        "name": "GPT-2 (tiktoken)",
        "vocab_size": 50257,
        "encode": tiktoken.get_encoding("gpt2").encode
    },
    "llama3": {
        "name": "Meta-Llama-3-8B (HF, 128k)",
        "vocab_size": 128000,
        "encode": lambda s, tok=AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B"): tok.encode(s, add_special_tokens=False)
    },
    "qwen2.5": {
        "name": "Qwen2.5-7B (HF, 152k)",
        "vocab_size": 151643,
        "encode": lambda s, tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"): tok.encode(s, add_special_tokens=False)
    },
    "xlm-roberta": {
        "name": "XLM-RoBERTa-base (HF, 250k)",
        "vocab_size": 250002,
        "encode": lambda s, tok=AutoTokenizer.from_pretrained("xlm-roberta-base"): tok.encode(s, add_special_tokens=False)
    }
}
print("All tokenizers loaded successfully.")


def run_experiment_A_whitespace(corpus_dict, corpus_name="sample"):
    """Test line.split(' ') vs line.split()"""
    enc = tokenizers["gpt2"]["encode"]
    results = []
    
    for lang, path in corpus_dict.items():
        lines = [unicodedata.normalize("NFC", l) for l in load_raw_lines(path)]
        
        # Original: line.split(" ") with lower()
        fert_orig_list = []
        # Corrected: line.split() with lower()
        fert_corr_list = []
        
        words_orig_total = 0
        words_corr_total = 0
        
        for l in lines:
            ll = l.lower()
            toks = enc(ll)
            w_orig = ll.split(" ")
            w_corr = ll.split()
            
            fert_orig_list.append(len(toks) / len(w_orig))
            fert_corr_list.append(len(toks) / len(w_corr))
            
            words_orig_total += len(w_orig)
            words_corr_total += len(w_corr)
            
        macro_orig = sum(fert_orig_list) / len(fert_orig_list)
        macro_corr = sum(fert_corr_list) / len(fert_corr_list)
        diff = macro_corr - macro_orig
        pct = (diff / macro_orig) * 100
        
        results.append({
            "corpus": corpus_name,
            "lang": lang,
            "macro_fert_orig_split_space": round(macro_orig, 4),
            "macro_fert_corr_split_ws": round(macro_corr, 4),
            "words_orig": words_orig_total,
            "words_corr": words_corr_total,
            "abs_diff": round(diff, 4),
            "pct_change": round(pct, 2)
        })
    return results


def run_experiment_B_lowercasing(corpus_dict, corpus_name="sample"):
    """Test line.lower() vs raw case across tokenizers"""
    results = []
    for tok_key in ["gpt2", "llama3"]:
        enc = tokenizers[tok_key]["encode"]
        for lang, path in corpus_dict.items():
            lines = [unicodedata.normalize("NFC", l) for l in load_raw_lines(path)]
            
            fert_lower = []
            fert_raw = []
            tpc_lower = []
            tpc_raw = []
            
            toks_lower_total = 0
            toks_raw_total = 0
            
            for l in lines:
                ll = l.lower()
                t_lower = enc(ll)
                t_raw = enc(l)
                
                w = l.split()
                c = len(l)
                
                fert_lower.append(len(t_lower) / len(w))
                fert_raw.append(len(t_raw) / len(w))
                tpc_lower.append(len(t_lower) / len(ll))
                tpc_raw.append(len(t_raw) / c)
                
                toks_lower_total += len(t_lower)
                toks_raw_total += len(t_raw)
                
            m_fl = sum(fert_lower) / len(fert_lower)
            m_fr = sum(fert_raw) / len(fert_raw)
            m_tpcl = sum(tpc_lower) / len(tpc_lower)
            m_tpcr = sum(tpc_raw) / len(tpc_raw)
            
            diff_f = m_fr - m_fl
            pct_f = (diff_f / m_fl) * 100
            diff_toks = toks_raw_total - toks_lower_total
            pct_toks = (diff_toks / toks_lower_total) * 100
            
            results.append({
                "corpus": corpus_name,
                "tokenizer": tok_key,
                "lang": lang,
                "fert_lower": round(m_fl, 4),
                "fert_cased": round(m_fr, 4),
                "fert_pct_change": round(pct_f, 2),
                "tok_per_char_lower": round(m_tpcl, 4),
                "tok_per_char_cased": round(m_tpcr, 4),
                "total_tokens_lower": toks_lower_total,
                "total_tokens_cased": toks_raw_total,
                "total_tokens_pct_change": round(pct_toks, 2)
            })
    return results


def run_experiment_C_aggregation(corpus_dict, corpus_name="flore200"):
    """Compare Macro-average (mean of ratios) vs Micro-average (ratio of sums)"""
    results = []
    for tok_key in ["gpt2", "llama3"]:
        enc = tokenizers[tok_key]["encode"]
        for lang, path in corpus_dict.items():
            lines = [unicodedata.normalize("NFC", l) for l in load_raw_lines(path)]
            
            per_line_fert = []
            per_line_tpc = []
            
            sum_tokens = 0
            sum_words = 0
            sum_chars = 0
            
            for l in lines:
                toks = enc(l)
                words = l.split()
                chars = len(l)
                
                per_line_fert.append(len(toks) / len(words))
                per_line_tpc.append(len(toks) / chars)
                
                sum_tokens += len(toks)
                sum_words += len(words)
                sum_chars += chars
                
            macro_fert = sum(per_line_fert) / len(per_line_fert)
            micro_fert = sum_tokens / sum_words
            
            macro_tpc = sum(per_line_tpc) / len(per_line_tpc)
            micro_tpc = sum_tokens / sum_chars
            
            diff_fert = micro_fert - macro_fert
            pct_fert = (diff_fert / macro_fert) * 100
            
            diff_tpc = micro_tpc - macro_tpc
            pct_tpc = (diff_tpc / macro_tpc) * 100
            
            results.append({
                "corpus": corpus_name,
                "tokenizer": tok_key,
                "lang": lang,
                "macro_fert": round(macro_fert, 4),
                "micro_fert": round(micro_fert, 4),
                "diff_fert": round(diff_fert, 4),
                "pct_fert": round(pct_fert, 2),
                "macro_tpc": round(macro_tpc, 4),
                "micro_tpc": round(micro_tpc, 4),
                "diff_tpc": round(diff_tpc, 4),
                "pct_tpc": round(pct_tpc, 2)
            })
    return results


def run_experiment_E_tokenizer_comparison(corpus_dict, corpus_name="flore200"):
    """Evaluate full cross-tokenizer efficiency across all 4 languages"""
    results = []
    
    # Store token counts to compute relative expansion vs English
    tokens_by_tok = {}
    
    for tok_key, tok_meta in tokenizers.items():
        enc = tok_meta["encode"]
        tokens_by_tok[tok_key] = {}
        
        for lang, path in corpus_dict.items():
            lines = [unicodedata.normalize("NFC", l) for l in load_raw_lines(path)]
            
            total_tokens = 0
            total_words = 0
            total_chars = 0
            total_bytes = 0
            
            per_line_fert = []
            per_line_tpc = []
            per_line_tpb = []
            
            for l in lines:
                toks = enc(l)
                words = l.split()
                chars = len(l)
                b_count = len(l.encode("utf-8"))
                
                n_t = len(toks)
                total_tokens += n_t
                total_words += len(words)
                total_chars += chars
                total_bytes += b_count
                
                per_line_fert.append(n_t / len(words))
                per_line_tpc.append(n_t / chars)
                per_line_tpb.append(n_t / b_count)
                
            macro_fert = sum(per_line_fert) / len(per_line_fert)
            micro_fert = total_tokens / total_words
            micro_tpc = total_tokens / total_chars
            micro_tpb = total_tokens / total_bytes
            
            tokens_by_tok[tok_key][lang] = total_tokens
            
            results.append({
                "corpus": corpus_name,
                "tokenizer_key": tok_key,
                "tokenizer_name": tok_meta["name"],
                "vocab_size": tok_meta["vocab_size"],
                "lang": lang,
                "total_tokens": total_tokens,
                "fertility_tok_per_word": round(micro_fert, 4),
                "macro_fertility": round(macro_fert, 4),
                "tok_per_char": round(micro_tpc, 4),
                "tok_per_utf8_byte": round(micro_tpb, 4),
                "total_words": total_words,
                "total_chars": total_chars,
                "total_bytes": total_bytes
            })
            
    # Calculate ratios relative to English
    for r in results:
        tkey = r["tokenizer_key"]
        eng_tokens = tokens_by_tok[tkey]["eng"]
        eng_fert = [x["fertility_tok_per_word"] for x in results if x["tokenizer_key"] == tkey and x["lang"] == "eng"][0]
        
        r["total_tokens_vs_eng_ratio"] = round(r["total_tokens"] / eng_tokens, 3)
        r["fertility_vs_eng_ratio"] = round(r["fertility_tok_per_word"] / eng_fert, 3)
        
    return results


def run_experiment_suspicious_normalization(corpus_dict, corpus_name="sample"):
    """Test NFC vs NFD vs Raw Unicode normalization"""
    enc = tokenizers["gpt2"]["encode"]
    results = []
    
    for lang, path in corpus_dict.items():
        raw_lines = load_raw_lines(path)
        
        toks_nfc = 0
        toks_nfd = 0
        toks_raw = 0
        
        for l in raw_lines:
            l_raw = l.lower()
            l_nfc = unicodedata.normalize("NFC", l_raw)
            l_nfd = unicodedata.normalize("NFD", l_raw)
            
            toks_raw += len(enc(l_raw))
            toks_nfc += len(enc(l_nfc))
            toks_nfd += len(enc(l_nfd))
            
        results.append({
            "corpus": corpus_name,
            "lang": lang,
            "tokens_nfc": toks_nfc,
            "tokens_raw": toks_raw,
            "tokens_nfd": toks_nfd,
            "nfd_inflation_pct": round(((toks_nfd - toks_nfc) / toks_nfc) * 100, 2)
        })
    return results


def main():
    print("=" * 80)
    print("RUNNING FORENSIC AUDIT OF fertility.py (PART A2)")
    print("=" * 80)
    
    # 1. Whitespace splitting experiment
    print("\n--- Running Experiment A: Whitespace Handling ---")
    res_A_sample = run_experiment_A_whitespace(SAMPLE_CORPUS, "sample")
    res_A_full = run_experiment_A_whitespace(FULL_CORPUS, "flore200")
    
    # 2. Lowercasing experiment
    print("\n--- Running Experiment B: Lowercasing ---")
    res_B_sample = run_experiment_B_lowercasing(SAMPLE_CORPUS, "sample")
    res_B_full = run_experiment_B_lowercasing(FULL_CORPUS, "flore200")
    
    # 3. Aggregation experiment
    print("\n--- Running Experiment C: Aggregation (Macro vs Micro) ---")
    res_C_full = run_experiment_C_aggregation(FULL_CORPUS, "flore200")
    
    # 4. Tokenizer scaling experiment
    print("\n--- Running Experiment E: Cross-Tokenizer & Conceptual Audit ---")
    res_E_sample = run_experiment_E_tokenizer_comparison(SAMPLE_CORPUS, "sample")
    res_E_full = run_experiment_E_tokenizer_comparison(FULL_CORPUS, "flore200")
    
    # 5. Suspicious-but-correct: Unicode normalization
    print("\n--- Running Suspicious-but-Correct Experiment: Unicode Normalization ---")
    res_norm_sample = run_experiment_suspicious_normalization(SAMPLE_CORPUS, "sample")
    res_norm_full = run_experiment_suspicious_normalization(FULL_CORPUS, "flore200")
    
    # Save all results to CSV / JSON
    with open(os.path.join(RESULTS_DIR, "exp_A_whitespace.json"), "w", encoding="utf-8") as f:
        json.dump({"sample": res_A_sample, "flore200": res_A_full}, f, indent=2)
        
    with open(os.path.join(RESULTS_DIR, "exp_B_lowercasing.json"), "w", encoding="utf-8") as f:
        json.dump({"sample": res_B_sample, "flore200": res_B_full}, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "exp_C_aggregation.json"), "w", encoding="utf-8") as f:
        json.dump(res_C_full, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "exp_E_tokenizers.json"), "w", encoding="utf-8") as f:
        json.dump({"sample": res_E_sample, "flore200": res_E_full}, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "exp_norm_unicode.json"), "w", encoding="utf-8") as f:
        json.dump({"sample": res_norm_sample, "flore200": res_norm_full}, f, indent=2)

    print("\n" + "=" * 80)
    print("AUDIT EXECUTION COMPLETE. Summary of Key Tables:")
    print("=" * 80)
    
    print("\n[TABLE 1: Cross-Tokenizer Comparison on FLORES-200 (997 Parallel Sentences)]")
    header1 = f"{'Tokenizer':<22}{'Lang':<6}{'Total Toks':>12}{'Tok/Word':>10}{'Tok/Char':>10}{'Tok/Byte':>10}{'Tokens vs Eng':>15}{'Fert vs Eng':>15}"
    print(header1)
    print("-" * len(header1))
    for r in res_E_full:
        print(f"{r['tokenizer_name']:<22}{r['lang']:<6}{r['total_tokens']:>12d}{r['fertility_tok_per_word']:>10.2f}{r['tok_per_char']:>10.3f}{r['tok_per_utf8_byte']:>10.3f}{r['total_tokens_vs_eng_ratio']:>14.2f}x{r['fertility_vs_eng_ratio']:>14.2f}x")
    print("-" * len(header1))

    print("\n[TABLE 2: Whitespace Handling (split(' ') vs split())]")
    for r in res_A_sample + res_A_full:
        print(f"[{r['corpus']}] {r['lang']}: split(' ')={r['macro_fert_orig_split_space']} -> split()={r['macro_fert_corr_split_ws']} (diff: {r['abs_diff']:+.4f}, {r['pct_change']:+.2f}%)")

    print("\n[TABLE 3: Lowercasing Impact (GPT-2 vs LLaMA-3)]")
    for r in res_B_full:
        print(f"[{r['tokenizer']}] {r['lang']}: Lower={r['total_tokens_lower']} -> Cased={r['total_tokens_cased']} (tokens delta: {r['total_tokens_pct_change']:+.2f}%) | Fert: {r['fert_lower']} -> {r['fert_cased']} ({r['fert_pct_change']:+.2f}%)")

    print("\n[TABLE 4: Suspicious-but-Correct: Unicode Normalization (NFC vs NFD)]")
    for r in res_norm_full:
        print(f"[{r['corpus']}] {r['lang']}: NFC Tokens={r['tokens_nfc']} | NFD Tokens={r['tokens_nfd']} | NFD Inflation: {r['nfd_inflation_pct']:+.2f}%")


if __name__ == "__main__":
    main()
