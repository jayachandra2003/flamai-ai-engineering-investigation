#!/usr/bin/env python3
"""
capacity_analysis.py -- Theoretical KV Cache Capacity & Hardware Modeling for Part B1.

Models the exact memory footprint of FLM-4B-Instruct on NVIDIA L4 (24GB)
under both decimal (GB) and binary (GiB) conventions.
"""

import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hardware & Model Specs from bench/model_spec.md
SPECS = {
    "model_name": "FLM-4B-Instruct",
    "params_billion": 4.2,
    "layers": 28,
    "d_model": 3072,
    "q_heads": 24,
    "kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 128000,
    "weights_precision": "fp16",  # 2 bytes/param
    "kv_precision": "fp16",       # 2 bytes/element
    "gpu_name": "NVIDIA L4",
    "vram_nominal_gb": 24.0,
    "memory_bandwidth_gb_s": 300.0,
    "fp16_dense_tflops": 121.0,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.92,
    "non_kv_overhead_gb": 1.6
}


def calculate_kv_capacity():
    bytes_per_param = 2  # fp16
    bytes_per_element = 2  # fp16
    seq_len = SPECS["max_model_len"]

    # KV bytes per token: 2 * layers * kv_heads * head_dim * bytes_per_element
    kv_bytes_per_token = 2 * SPECS["layers"] * SPECS["kv_heads"] * SPECS["head_dim"] * bytes_per_element
    kv_bytes_per_seq = seq_len * kv_bytes_per_token

    # -------------------------------------------------------------
    # 1. Decimal Convention (1 GB = 10^9 bytes)
    # -------------------------------------------------------------
    vram_dec_total = SPECS["vram_nominal_gb"] * 1e9             # 24.0 GB
    usable_dec_vram = vram_dec_total * SPECS["gpu_memory_utilization"]  # 22.08 GB
    weights_dec_bytes = SPECS["params_billion"] * 1e9 * bytes_per_param # 8.40 GB
    overhead_dec_bytes = SPECS["non_kv_overhead_gb"] * 1e9              # 1.60 GB
    kv_budget_dec_bytes = usable_dec_vram - weights_dec_bytes - overhead_dec_bytes # 12.08 GB

    max_seqs_decimal = kv_budget_dec_bytes / kv_bytes_per_seq # 12.08e9 / 469,762,048 = 25.71

    # Theoretical utilization at batch 24 under decimal budget
    b24_demand_bytes = 24 * kv_bytes_per_seq # 11,274,289,152 bytes (11.27 GB)
    b24_util_decimal = (b24_demand_bytes / kv_budget_dec_bytes) # 0.9333 (93.3%)

    # -------------------------------------------------------------
    # 2. Binary Convention (1 GiB = 1024^3 bytes)
    # -------------------------------------------------------------
    vram_bin_total = 24.0 * (1024**3)                           # 25,769,803,776 bytes
    usable_bin_vram = vram_bin_total * SPECS["gpu_memory_utilization"] # 22.08 GiB
    weights_bin_bytes = weights_dec_bytes                       # 8.4e9 bytes = 7.823 GiB
    overhead_bin_bytes = SPECS["non_kv_overhead_gb"] * (1024**3) # 1.6 GiB
    kv_budget_bin_bytes = usable_bin_vram - weights_bin_bytes - overhead_bin_bytes # 12.657 GiB

    max_seqs_binary = kv_budget_bin_bytes / kv_bytes_per_seq    # 28.93

    results = {
        "hardware_and_model_specs": SPECS,
        "token_and_sequence_footprint": {
            "kv_bytes_per_token": kv_bytes_per_token,
            "kv_kib_per_token": kv_bytes_per_token / 1024,
            "seq_len": seq_len,
            "kv_bytes_per_4096_seq": kv_bytes_per_seq,
            "kv_mib_per_4096_seq": kv_bytes_per_seq / (1024**2),
            "kv_decimal_gb_per_4096_seq": round(kv_bytes_per_seq / 1e9, 4)
        },
        "decimal_gb_modeling": {
            "usable_vram_gb": round(usable_dec_vram / 1e9, 2),
            "model_weights_gb": round(weights_dec_bytes / 1e9, 2),
            "non_kv_overhead_gb": round(overhead_dec_bytes / 1e9, 2),
            "available_kv_budget_gb": round(kv_budget_dec_bytes / 1e9, 2),
            "theoretical_max_4096_seqs": round(max_seqs_decimal, 2),
            "batch_24_demand_gb": round(b24_demand_bytes / 1e9, 2),
            "batch_24_predicted_utilization": round(b24_util_decimal, 4)
        },
        "binary_gib_modeling": {
            "usable_vram_gib": round(usable_bin_vram / (1024**3), 3),
            "model_weights_gib": round(weights_bin_bytes / (1024**3), 3),
            "non_kv_overhead_gib": round(overhead_bin_bytes / (1024**3), 3),
            "available_kv_budget_gib": round(kv_budget_bin_bytes / (1024**3), 3),
            "theoretical_max_4096_seqs": round(max_seqs_binary, 2)
        }
    }
    return results


def main():
    res = calculate_kv_capacity()
    out_path = os.path.join(RESULTS_DIR, "kv_capacity_derivation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    t = res["token_and_sequence_footprint"]
    dec = res["decimal_gb_modeling"]
    bi = res["binary_gib_modeling"]

    print("=" * 80)
    print("PART B1: THEORETICAL KV CACHE CAPACITY & RECONCILIATION")
    print("=" * 80)
    print(f"1. KV bytes per token: {t['kv_bytes_per_token']} bytes ({t['kv_kib_per_token']} KiB/token)")
    print(f"2. Memory per 4096-token sequence: {t['kv_bytes_per_4096_seq']:,} bytes ({t['kv_mib_per_4096_seq']:.1f} MiB = {t['kv_decimal_gb_per_4096_seq']} GB)")
    print("-" * 80)
    print("Decimal GB Modeling (Direct match to logged 0.93 utilization):")
    print(f"  Usable VRAM: {dec['usable_vram_gb']} GB - Weights: {dec['model_weights_gb']} GB - Overhead: {dec['non_kv_overhead_gb']} GB = KV Budget: {dec['available_kv_budget_gb']} GB")
    print(f"  Theoretical Max Concurrency: {dec['available_kv_budget_gb']} GB / {t['kv_decimal_gb_per_4096_seq']} GB = {dec['theoretical_max_4096_seqs']} sequences")
    print(f"  Batch 24 Demand: 24 * {t['kv_decimal_gb_per_4096_seq']} GB = {dec['batch_24_demand_gb']} GB -> Predicted Util: {dec['batch_24_predicted_utilization']*100:.1f}% (Matches logged 0.93!)")
    print("-" * 80)
    print(f"Binary GiB Modeling: Available KV = {bi['available_kv_budget_gib']} GiB -> Theoretical Max = {bi['theoretical_max_4096_seqs']} sequences")
    print("=" * 80)


if __name__ == "__main__":
    main()
