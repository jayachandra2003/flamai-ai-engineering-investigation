# FlamAI Engineering Investigation: The Audit

## Overview

This repository contains the complete forensic audit and corrected engineering analysis for the FlamAI AI team. The investigation independently evaluates tokenizer fertility, multilingual routing metrics, serving hardware capacity, and tone optimization strategies across three core areas:

* **Part A (Tokenizer & Fertility Audit):** Re-evaluating cross-lingual token expansion across English, Hindi, Kannada, and Telugu on a 997-sentence parallel FLORES-200 benchmark.
* **Part B (Serving Forensics & KV Cache Capacity):** Reconciling theoretical memory limits on 1× NVIDIA L4 (24 GB) with empirical load test logs in `bench/bench_log.csv`.
* **Part C (Strategic Decision Analysis):** Determining the optimal technical strategy (SFT vs. $\le 1\text{B}$ Rewriter vs. Prompt Engineering) to achieve conversational tone across 6 Indic languages under strict compute and human reviewer constraints.

---

## Repository Structure

```text
.
├── README.md                              # Investigation overview & reproduction guide
├── NOTEBOOK.md                            # Comprehensive chronological investigation audit trail
├── AI_USAGE.md                            # AI tools disclosure and division of work
├── REPORT_v0.md                           # Original intern draft (unmodified baseline)
├── fertility.py                           # Original baseline script (unmodified)
├── bench/
│   ├── bench_log.csv                      # Raw load test benchmark log (unmodified)
│   └── model_spec.md                      # FLM-4B-Instruct serving specification (unmodified)
├── corpus_sample/
│   ├── eng_sample.txt                     # Starter English sample (unmodified)
│   └── hin_sample.txt                     # Starter Hindi sample (unmodified)
├── partA/
│   ├── README.md                          # Corpus documentation and limitations
│   ├── recommendation_memo.md             # Executive one-page multilingual routing memo
│   ├── corpus/                            # 997 parallel FLORES-200 sentences (eng, hin, kan, tel)
│   ├── scripts/
│   │   └── prepare_corpus.py              # Automated dataset fetch & alignment verification
│   ├── audit/
│   │   ├── README.md                      # Detailed forensic audit of fertility.py
│   │   ├── run_audit.py                   # Automated multi-hypothesis audit test suite
│   │   └── results/                       # Raw JSON experiment logs
│   └── corrected_analysis/
│       ├── README.md                      # Denominator proofs & metric comparison
│       ├── analyze_metrics.py             # 4-tokenizer cross-lingual metric analysis script
│       └── results/                       # JSON metrics for GPT-2, LLaMA-3, Qwen2.5, XLM-R
├── partB/
│   ├── README.md                          # Theoretical KV capacity & throughput reconciliation
│   └── analysis/
│       ├── capacity_analysis.py           # Mathematical KV cache hardware model
│       ├── benchmark_analysis.py          # Forensic breakdown of bench_log.csv
│       └── results/                       # Mathematical derivation and goodput JSON logs
└── partC/
    ├── memo.md                            # Strategic AI decision memo for leadership
    └── analysis_scratch.py                # Reviewer and compute arithmetic verification script
```

*Note: All original starter-kit files remain 100% untouched and preserved at their original baseline states.*

---

## Key Findings

1. **Tokenizer Mismatch Invalidates Script Invariance Claims:** `REPORT_v0.md` tested only the English-centric `gpt2` tokenizer (50k vocabulary) and claimed high fertility is an inherent property of the Devanagari script. On our 997-sentence parallel benchmark, Hindi token expansion drops from **7.45× on GPT-2** to **2.53× on LLaMA-3 (128k)** and **1.26× on XLM-RoBERTa (250k)**, strongly demonstrating that token expansion is tokenizer-dependent.
2. **Whitespace-Word Fertility Distorts Dravidian Comparisons:** In agglutinative languages like Kannada, whitespace-word fertility ($\text{tok/word}$) artificially inflates apparent inefficiency by **+35.8%** because morphemes fuse into fewer words.
3. **Primary Routing Metric Decision:** **Relative Token Expansion Ratio on Parallel Evaluation Data** ($\frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}}$) was established as the primary routing/cost metric because inference compute, KV cache allocation, and token pricing scale with total token sequence length.
4. **Empirical Safe Operating Point at Batch 24:** Batch 24 was the highest tested configuration with zero preemptions. The theoretical 4096-token capacity is approximately 25–28 sequences; batch 32 exceeded the available KV-cache budget and triggered 7 preemptions. At batch 48, severe memory saturation triggers **23 sequence preemptions**, cutting throughput to $1298.5\text{ tok/s}$ and inflating p95 latency to $105.4\text{ seconds}$.
5. **Prompt Engineering Is the Optimal Day-1 Strategy:** Under the constraints of 1 A100 GPU for 2 weeks, a single native reviewer (30 hours total covering Hindi/Kannada only), and a 3-week launch window, Prompt Engineering provides immediate empirical feedback on Day 1 without consuming upfront GPU training time, preserving A100 compute for SFT if a pivot is required at Day 7.

---

## Reproducibility Guide

All scripts are completely self-contained and executable using standard Python 3.11+:

```powershell
# 1. Prepare and validate the parallel multilingual evaluation corpus (Part A1)
python partA/scripts/prepare_corpus.py

# 2. Run the multi-hypothesis forensic audit suite (Part A2)
python partA/audit/run_audit.py

# 3. Execute the corrected cross-tokenizer metric analysis (Part A3)
python partA/corrected_analysis/analyze_metrics.py

# 4. Run the theoretical KV cache capacity derivation (Part B1)
python partB/analysis/capacity_analysis.py

# 5. Run the forensic load test benchmark analysis (Part B2 & B3)
python partB/analysis/benchmark_analysis.py

# 6. Verify the Part C resource and reviewer arithmetic
python partC/analysis_scratch.py
```

---

## Limitations

1. **Proprietary FLM-4B Tokenizer:** The exact tokenizer for `FLM-4B-Instruct` was not provided in the starter kit. Evaluated public models (`Meta-Llama-3-8B`, `Qwen2.5-7B`, `XLM-RoBERTa-base`) illustrate vocabulary scaling behavior but do not represent exact FLM-4B production values.
2. **Offline Token Expansion vs. Live Serving Latency:** Offline token counts establish sequence length multipliers; live serving latency (TTFT, ITL, p95 E2E) is heavily influenced by prompt-to-generation token ratios, batch scheduler dynamics, and hardware memory bandwidth.
3. **Language Reviewer Coverage:** Direct native-speaker validation is strictly available for Hindi and Kannada (30 total reviewer-hours). Tamil, Telugu, Bengali, and Marathi lack direct native review under the current team configuration.
