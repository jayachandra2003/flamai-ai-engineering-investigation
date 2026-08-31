#!/usr/bin/env python3
"""
benchmark_analysis.py -- Forensic Analysis of bench_log.csv for Part B2, B3, B4.

Audits:
  - Throughput anomaly in prompt=3584 long-context sweep
  - Misread column in REPORT_v0.md
  - Exact wall-clock goodput vs decode-phase rate for Batch 24
  - Recommended production validation metric
"""

import os
import sys
import csv
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(BASE_DIR, "bench", "bench_log.csv")
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_analyze_bench():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    rows = []
    for r in reader:
        b = int(r["batch_size"])
        p = int(r["prompt_len"])
        g = int(r["gen_len"])
        n = int(r["num_requests"])
        w = float(r["wall_clock_s"])
        rep = float(r["reported_tok_s"])
        ttft = float(r["ttft_ms_p50"])
        itl = float(r["itl_ms_p50"])
        e2e = float(r["e2e_ms_p95"])
        preempt = int(r["preempted_seqs"])
        kv = float(r["kv_cache_util"])

        total_tokens = n * (p + g)
        gen_tokens = n * g
        prompt_tokens = n * p

        calc_total_tok_s = total_tokens / w
        calc_gen_tok_s = gen_tokens / w

        # Decode-phase instantaneous generation rate across batch
        itl_sec = itl / 1000.0
        decode_rate_tok_s = (b / itl_sec) if itl_sec > 0 else 0

        rows.append({
            "batch_size": b,
            "prompt_len": p,
            "gen_len": g,
            "total_seq_len": p + g,
            "num_requests": n,
            "wall_clock_s": w,
            "reported_tok_s": rep,
            "calc_total_tok_s": round(calc_total_tok_s, 2),
            "calc_gen_tok_s": round(calc_gen_tok_s, 2),
            "decode_rate_tok_s": round(decode_rate_tok_s, 2),
            "prompt_token_share_pct": round((p / (p + g)) * 100, 1),
            "ttft_ms_p50": ttft,
            "itl_ms_p50": itl,
            "e2e_ms_p95": e2e,
            "preempted_seqs": preempt,
            "kv_cache_util": kv
        })

    short_sweep = [r for r in rows if r["prompt_len"] == 512]
    long_sweep = [r for r in rows if r["prompt_len"] == 3584]

    # Batch 24 Analysis (P=3584, G=512, N=24, W=61.16s, ITL=96.07ms)
    b24_row = [r for r in long_sweep if r["batch_size"] == 24][0]
    
    # 1. Exact End-to-End Wall-Clock Goodput
    g_tokens_b24 = b24_row["num_requests"] * b24_row["gen_len"]  # 24 * 512 = 12,288 tokens
    exact_wall_clock_goodput = g_tokens_b24 / b24_row["wall_clock_s"]  # 12,288 / 61.16 = 200.92 tok/s

    # 2. Decode-Phase Generation Rate from ITL
    itl_sec_b24 = b24_row["itl_ms_p50"] / 1000.0                # 0.09607 s
    decode_phase_rate = b24_row["batch_size"] / itl_sec_b24       # 24 / 0.09607 = 249.82 tok/s

    results = {
        "short_prompt_sweep_512_256": short_sweep,
        "long_prompt_sweep_3584_512": long_sweep,
        "batch_24_goodput_analysis": {
            "batch_size": 24,
            "prompt_len": 3584,
            "gen_len": 512,
            "wall_clock_s": 61.16,
            "reported_tok_s": 1607.4,
            "exact_wall_clock_goodput_tok_s": round(exact_wall_clock_goodput, 2),
            "exact_wall_clock_goodput_formula": "(num_requests * gen_len) / wall_clock_s = (24 * 512) / 61.16",
            "decode_phase_rate_tok_s": round(decode_phase_rate, 2),
            "decode_phase_rate_formula": "batch_size / (itl_ms_p50 / 1000) = 24 / 0.09607",
            "why_decode_rate_differs_from_goodput": "Decode-phase rate (249.82 tok/s) measures the instantaneous token generation speed during active decode steps. End-to-end goodput (200.92 tok/s) is lower because the total wall-clock time (61.16s) also includes the prompt prefill phase (~500ms TTFT) and runtime/tail scheduling overhead.",
            "why_second_exact_e2e_derivation_is_not_possible": "The CSV supplies aggregate wall-clock time and median per-step latency (p50 TTFT, p50 ITL), but does not provide separate logged timers for total prefill wall-clock duration versus total decode wall-clock duration. Therefore, only one exact wall-clock goodput value (200.92 tok/s) can be directly calculated from the provided data."
        },
        "misread_column_audit": {
            "misread_column_name": "reported_tok_s",
            "definition": "reported_tok_s = (num_requests * (prompt_len + gen_len)) / wall_clock_s",
            "report_v0_interpretation": "Interpreted reported_tok_s as generation serving throughput and claimed longer prompts yield superior throughput (1311 vs 883 tok/s at batch 16). Projected linear scaling to ~3200 tok/s at batch 48.",
            "correct_interpretation": "reported_tok_s measures total processed tokens (prefill + decode) per wall-clock second. For long prompts, 87.5% of tokens are prompt tokens processed during parallel prefill. Generation goodput is actually 44.3% lower for long prompts (163.9 vs 294.5 tok/s at batch 16). At batch 48, memory exhaustion causes 23 preemptions, dropping reported throughput to 1298.5 tok/s."
        }
    }
    return results


def main():
    res = load_and_analyze_bench()
    out_path = os.path.join(RESULTS_DIR, "benchmark_forensics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("=" * 90)
    print("BENCHMARK FORENSICS (bench_log.csv)")
    print("=" * 90)
    b24 = res["batch_24_goodput_analysis"]
    print(f"Batch 24 Exact Wall-Clock Goodput: {b24['exact_wall_clock_goodput_tok_s']} tok/s ({b24['exact_wall_clock_goodput_formula']})")
    print(f"Batch 24 Decode-Phase Rate (from ITL): {b24['decode_phase_rate_tok_s']} tok/s ({b24['decode_phase_rate_formula']})")
    print(f"Explanation: {b24['why_decode_rate_differs_from_goodput']}")


if __name__ == "__main__":
    main()
