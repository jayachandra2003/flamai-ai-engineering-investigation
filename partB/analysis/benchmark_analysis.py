#!/usr/bin/env python3
"""
benchmark_analysis.py -- Forensic Analysis of bench_log.csv for Part B2, B3, B4.

Audits:
  - Throughput anomaly in prompt=3584 long-context sweep
  - Misread column in REPORT_v0.md
  - Two independent goodput derivations for Batch 24
  - Median decode-phase rate estimate
  - Single recommended production validation metric
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

    # Batch 24 Analysis (P=3584, G=512, N=24, W=61.16s, ITL=96.07ms, Reported=1607.4 tok/s)
    b24_row = [r for r in long_sweep if r["batch_size"] == 24][0]
    
    # -------------------------------------------------------------
    # Method 1: Generated Output Tokens / Wall Clock Time
    # -------------------------------------------------------------
    gen_tokens_b24 = b24_row["num_requests"] * b24_row["gen_len"]  # 24 * 512 = 12,288
    goodput_method1 = gen_tokens_b24 / b24_row["wall_clock_s"]      # 12,288 / 61.16 = 200.9156...

    # -------------------------------------------------------------
    # Method 2: Reported Throughput * Generated-Token Fraction
    # -------------------------------------------------------------
    gen_fraction = b24_row["gen_len"] / (b24_row["prompt_len"] + b24_row["gen_len"])  # 512 / 4096 = 0.125
    goodput_method2 = b24_row["reported_tok_s"] * gen_fraction                         # 1607.4 * 0.125 = 200.925

    # -------------------------------------------------------------
    # Separate Diagnostic: Median Decode-Phase Rate Estimate (from ITL)
    # -------------------------------------------------------------
    itl_sec_b24 = b24_row["itl_ms_p50"] / 1000.0                # 0.09607 s
    decode_phase_rate = b24_row["batch_size"] / itl_sec_b24       # 24 / 0.09607 = 249.82 tok/s

    results = {
        "short_prompt_sweep_512_256": short_sweep,
        "long_prompt_sweep_3584_512": long_sweep,
        "batch_24_goodput_analysis": {
            "batch_size": 24,
            "prompt_len": 3584,
            "gen_len": 512,
            "total_seq_len": 4096,
            "wall_clock_s": 61.16,
            "reported_tok_s": 1607.4,
            "method1_direct_generation_volume": {
                "formula": "(num_requests * gen_len) / wall_clock_s",
                "calculation": "(24 * 512) / 61.16",
                "result_tok_s": round(goodput_method1, 2),
                "exact_unrounded": goodput_method1
            },
            "method2_reported_throughput_scaling": {
                "formula": "reported_tok_s * (gen_len / (prompt_len + gen_len))",
                "calculation": "1607.4 * (512 / 4096)",
                "result_tok_s": round(goodput_method2, 2),
                "exact_unrounded": goodput_method2,
                "rounding_explanation": "The 0.01 tok/s difference (200.916 vs 200.925) arises from the 1-decimal rounding of reported_tok_s (1607.4 vs true 1607.325)."
            },
            "separate_decode_phase_rate_estimate": {
                "metric_name": "median decode-phase rate estimate",
                "formula": "batch_size / (itl_ms_p50 / 1000)",
                "calculation": "24 / 0.09607",
                "result_tok_s": round(decode_phase_rate, 2),
                "why_distinct_from_goodput": "This measures instantaneous generation speed during the memory-bound decode loop only, excluding prefill time (~500ms TTFT) and engine overhead."
            }
        },
        "misread_column_audit": {
            "misread_column_name": "reported_tok_s",
            "definition": "reported_tok_s = (num_requests * (prompt_len + gen_len)) / wall_clock_s",
            "report_v0_misinterpretation": "Treated reported_tok_s as generation serving throughput; concluded long prompts give higher throughput (1311 vs 883 tok/s) and projected ~3200 tok/s at batch 48.",
            "correct_interpretation": "reported_tok_s includes compute-bound parallel prefill tokens (87.5% of total). True generation goodput is 44.3% lower for long prompts (163.9 vs 294.5 tok/s). Batch 48 memory exhaustion causes 23 preemptions, collapsing throughput to 1298.5 tok/s."
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
    print(f"Batch 24 Goodput Method 1: {b24['method1_direct_generation_volume']['result_tok_s']} tok/s ({b24['method1_direct_generation_volume']['calculation']})")
    print(f"Batch 24 Goodput Method 2: {b24['method2_reported_throughput_scaling']['result_tok_s']} tok/s ({b24['method2_reported_throughput_scaling']['calculation']})")
    print(f"Separate Median Decode-Phase Rate: {b24['separate_decode_phase_rate_estimate']['result_tok_s']} tok/s ({b24['separate_decode_phase_rate_estimate']['calculation']})")


if __name__ == "__main__":
    main()
