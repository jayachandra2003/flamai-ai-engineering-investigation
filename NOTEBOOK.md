# Investigation Notebook

## Executive Summary

This investigation conducted a comprehensive forensic audit of the claims, measurement scripts, and capacity assumptions in `REPORT_v0.md` for the FlamAI AI engineering team.

### What Was Investigated
1. **Tokenizer Fertility & Cross-Lingual Cost Claims (Part A):** Auditing the claim that Hindi text is inherently 5.89×–7.0× more expensive to serve than English due to Unicode properties of the Devanagari script.
2. **Serving Throughput & KV Cache Capacity (Part B):** Auditing the claim that longer prompts universally improve serving throughput, and evaluating the recommendation to scale FLM-4B concurrency linearly to Batch 48 (~3200 tok/s) on an NVIDIA L4 GPU.
3. **Conversational Tone Strategy (Part C):** Evaluating whether Supervised Fine-Tuning (SFT), an auxiliary $\le 1\text{B}$ rewriter, or Prompt Engineering is the optimal technical path to achieve casual/conversational responses across 6 Indic languages within tight compute, human reviewer, and timeline constraints.

### Major Initial Assumptions Corrected
* **Tokenizer Generalization Flaw:** `REPORT_v0.md` tested only the English-centric `gpt2` tokenizer (50k vocabulary) from 2019 and attributed high fertility to the Devanagari script. Evaluating modern multilingual tokenizers on 997 parallel FLORES-200 sentences demonstrated that token expansion drops to **2.53× on LLaMA-3 (128k)** and **1.26× on XLM-RoBERTa (250k)**, proving token expansion is strongly tokenizer-dependent.
* **Morphological Distortion in Whitespace Fertility:** In agglutinative languages like Kannada, whitespace-word fertility ($\text{tok/word}$) artificially inflates apparent inefficiency by **+35.8%** because morphemes fuse into fewer words. Relative Token Expansion on parallel content ($\frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}}$) was established as the primary cost/routing metric.
* **Misread Throughput Column:** `reported_tok_s` was misread as generation throughput, when it actually measures total processed tokens (including prefill). At batch 16, actual generation goodput is **44.3% lower** for long prompts ($163.9\text{ tok/s}$) than short prompts ($294.5\text{ tok/s}$).
* **Physical Memory & Preemption Limits:** Linear scaling to batch 48 ignores physical GPU memory limits. The KV-cache saturation boundary is approached at batch 24; preemption begins at batch 32. At batch 48, 23 sequences are preempted, causing throughput to drop to **1298.5 tok/s** (not 3200 tok/s) and p95 latency to exceed **105 seconds**.

---

## Part A — Tokenizer & Fertility Audit

### A1 — Corpus Preparation
* **Corpus Selection:** Evaluated multi-way parallel corpora and selected **FLORES-200** (Meta AI / NLLB Team) `dev` split.
* **Language Representation:** 4 languages covering two major families:
  * **English (`eng_Latn`):** Germanic / Indo-European baseline.
  * **Hindi (`hin_Deva`):** Indo-Aryan / Indo-European.
  * **Kannada (`kan_Knda`):** South Dravidian.
  * **Telugu (`tel_Telu`):** South-Central Dravidian.
* **Dataset Scale & Alignment:** Exactly **997 parallel sentences** per language, canonicalized with Unicode NFC (`unicodedata.normalize('NFC', line)`), single sentence per line, UTF-8 encoded.
* **Audit Corrections During Investigation:**
  1. *Source Article Count:* Corrected documentation from "842 distinct articles" (which describes the entire FLORES-200 benchmark including hidden test) to **281 distinct article URLs** verified in `metadata_dev.tsv`.
  2. *Download Mechanism:* Corrected description from "streaming HTTP GET" to an in-memory buffered fetch (`requests.get(...)` into `io.BytesIO`).
* **Ground Truth Corpus Statistics (FLORES-200 dev):**
  * `eng`: 997 lines | 20,954 whitespace words | 125,194 chars | 125,290 UTF-8 text bytes
  * `hin`: 997 lines | 24,607 whitespace words | 125,495 chars | 322,640 UTF-8 text bytes
  * `kan`: 997 lines | 15,430 whitespace words | 131,749 chars | 357,408 UTF-8 text bytes
  * `tel`: 997 lines | 16,388 whitespace words | 127,172 chars | 338,804 UTF-8 text bytes

---

### A2 — Forensic Audit of `fertility.py`

Controlled experiments were executed across starter samples and FLORES-200 using `partA/audit/run_audit.py`:

| Hypothesis / Issue | Implementation Location | Before Value (Observed) | After Value (Corrected) | Delta / Change (%) | Audit Finding & Status |
|---|---|:---:|:---:|:---:|---|
| **Whitespace Splitting** | `fertility.py:62`<br>`line.split(" ")` vs `split()` | Starter eng: `1.2652`<br>Starter hin: `7.4485` | Starter eng: `1.2831`<br>Starter hin: `7.5985` | eng: **+1.41%**<br>hin: **+2.01%** | **CONFIRMED IMPLEMENTATION FLAW:** `split(" ")` creates empty strings on multiple spaces. On clean FLORES-200, effect is minor (+0.0% to +0.04% for eng/hin; +3.6% for kan). |
| **Lowercasing Asymmetry** | `fertility.py:60`<br>`line.lower()` vs raw case | FLORES eng: 26,696 tok<br>FLORES hin: 191,842 tok | FLORES eng: 25,741 tok<br>FLORES hin: 191,828 tok | eng: **-3.58%**<br>hin: **-0.01%** | **CONFIRMED IMPLEMENTATION ASYMMETRY:** Devanagari has no casing (0.0% effect), but lowercasing fractures English uppercase acronyms (`NASA`, `GPU`), slightly inflating English token counts. |
| **Macro vs Micro Aggregation** | `fertility.py:67`<br>Mean of ratios vs Ratio of sums | LLaMA-3 eng: `1.2395`<br>LLaMA-3 hin: `2.6667` | LLaMA-3 eng: `1.2309`<br>LLaMA-3 hin: `2.6562` | eng: **-0.70%**<br>hin: **-0.39%** | **CONFIRMED MINOR DIFFERENCE:** Micro-average is mathematically preferable for total system volume; divergence on sentence data is $<1.3\%$. |
| **Tokenizer Model Mismatch** | `fertility.py:79`<br>GPT-2 vs Modern Multilingual | GPT-2 hin: 191,828 tok (7.45× eng) | LLaMA-3: 65,361 tok (2.53× eng)<br>XLM-R: 36,634 tok (1.26× eng) | LLaMA-3: **-65.9%**<br>XLM-R: **-80.9%** | **CONFIRMED MAJOR METHODOLOGICAL FLAW:** Hindi token expansion is strongly tokenizer-dependent and cannot be attributed solely to the script. |
| **Unicode NFC Normalization (Suspicious-but-Correct)** | `fertility.py:49`<br>NFC vs NFD normalization | FLORES kan (NFC): 349,802 tok | FLORES kan (NFD): 364,303 tok | kan: **+4.15% inflation** | **CONFIRMED CORRECT:** In Indic scripts, decomposed NFD matras fragment into byte-fallback tokens. NFC canonical composition is standard, essential NLP preprocessing. |

---

### A3 — Corrected Multilingual Metric Analysis

Controlled evaluation on 997 parallel FLORES-200 sentences across 4 tokenizers via `partA/corrected_analysis/analyze_metrics.py`:

| Tokenizer | Vocab Size | Language | Total Tokens | Tok / Word | Tok / Char | Tok / UTF-8 Byte | Relative Token Expansion vs. Eng | Fertility Ratio vs. Eng |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPT-2** (tiktoken) | 50,257 | **eng** | 25,741 | 1.23 | 0.206 | 0.205 | 1.00× | 1.00× |
| | | **hin** | 191,828 | 7.80 | 1.529 | 0.595 | **7.45×** | **6.35×** |
| | | **kan** | 349,772 | 22.67 | 2.655 | 0.979 | **13.59×** | **18.45×** |
| | | **tel** | 335,642 | 20.48 | 2.639 | 0.991 | **13.04×** | **16.67×** |
| **Meta-Llama-3-8B** (128k scale) | 128,000 | **eng** | 25,792 | 1.23 | 0.206 | 0.206 | 1.00× | 1.00× |
| | | **hin** | 65,361 | 2.66 | 0.521 | 0.203 | **2.53×** | **2.16×** |
| | | **kan** | 229,014 | 14.84 | 1.738 | 0.641 | **8.88×** | **12.06×** |
| | | **tel** | 215,433 | 13.15 | 1.694 | 0.636 | **8.35×** | **10.68×** |
| **Qwen2.5-7B** (152k scale) | 151,643 | **eng** | 26,255 | 1.25 | 0.210 | 0.210 | 1.00× | 1.00× |
| | | **hin** | 116,701 | 4.74 | 0.930 | 0.362 | **4.45×** | **3.79×** |
| | | **kan** | 182,074 | 11.80 | 1.382 | 0.509 | **6.93×** | **9.42×** |
| | | **tel** | 185,113 | 11.30 | 1.456 | 0.546 | **7.05×** | **9.02×** |
| **XLM-RoBERTa-base** (250k scale) | 250,002 | **eng** | 28,995 | 1.38 | 0.232 | 0.231 | 1.00× | 1.00× |
| | | **hin** | 36,634 | 1.49 | 0.292 | 0.114 | **1.26×** | **1.08×** |
| | | **kan** | 39,602 | 2.57 | 0.301 | 0.111 | **1.37×** | **1.85×** |
| | | **tel** | 38,708 | 2.36 | 0.304 | 0.114 | **1.33×** | **1.71×** |

#### Why Whitespace Fertility Was Rejected as Primary Metric:
$$\text{Fertility Ratio} = \left(\frac{T_{\text{lang}}}{T_{\text{eng}}}\right) \times \left(\frac{W_{\text{eng}}}{W_{\text{lang}}}\right)$$
Because Kannada is agglutinative, expressing 997 sentences in only 15,430 words vs. English's 20,954 words ($\frac{W_{\text{eng}}}{W_{\text{kan}}} = \mathbf{1.358}$), whitespace fertility inflates apparent inefficiency by **+35.8%**. On XLM-R, Kannada fertility is 1.85× English, but its actual token footprint is only **1.37×** English!

* **Selected Primary Routing/Cost Metric:** **Relative Token Expansion Ratio on Parallel Evaluation Data** ($\frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}}$).
* **Selected Secondary Diagnostic Metric:** **Tokens per UTF-8 Byte** ($\text{Tok} / \text{Byte}$) relative to encoded representation.

---

### A4 — Recommendation Summary
* **Recommendation Memo:** Synthesized in `partA/recommendation_memo.md`.
* **Headline Advice:** Base cost models on Relative Token Expansion on parallel corpora rather than whitespace-word fertility. Do not budget a 6× serving cost multiplier or separate Indic routing infrastructure based on GPT-2 benchmarks.
* **Production Metric to Monitor:** **Mean Total Tokens per Request by Language relative to English** ($\frac{\bar{T}_{\text{lang, req}}}{\bar{T}_{\text{eng, req}}}$).

---

## Part B — Serving Forensics

### B1 — KV Cache Mathematical Modeling
* **Model Parameters:** `FLM-4B-Instruct` (4.2B params, 28 layers, 8 KV heads GQA, head dim 128, fp16).
* **Hardware:** 1× NVIDIA L4 (24 GB VRAM, 300 GB/s bandwidth).
* **KV Bytes per Token:**
  $$\text{KV bytes/token} = 2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688\text{ bytes}} = \mathbf{112.0\text{ KiB/token}}$$
* **KV Memory per 4096-Token Sequence:**
  $$4096 \times 114,688 = \mathbf{469,762,048\text{ bytes}} = \mathbf{448.0\text{ MiB}} \approx \mathbf{0.4698\text{ GB}}$$
* **Available KV Memory & Theoretical Concurrency Ceiling:**
  * Usable VRAM: $24.0\text{ GB} \times 0.92 = 22.08\text{ GB}$
  * Minus Model Weights: $4.2\text{B} \times 2 = 8.40\text{ GB}$
  * Minus Runtime Overhead: $1.60\text{ GB}$
  * Available KV Cache Memory: $22.08 - 8.40 - 1.60 = \mathbf{12.08\text{ GB}}$ ($12,960.5\text{ MiB}$ in binary GiB)
  * **Theoretical Maximum Concurrency (4096 tokens):** $\frac{12.08\text{ GB}}{0.4698\text{ GB}} = \mathbf{25.7\text{ sequences}}$ (decimal) to $\mathbf{28.9\text{ sequences}}$ (binary).
* **Reconciling the 0.93 Logged Utilization at Batch 24:**
  At batch 24, memory demand is $24 \times 0.4698\text{ GB} = 11.274\text{ GB}$. Expressed as a fraction of the allocated $12.08\text{ GB}$ pool: $\frac{11.274}{12.080} = \mathbf{0.933} \approx \mathbf{0.93}$.

---

### B2 — Throughput Anomaly Analysis

From `bench/bench_log.csv` (Prompt=3584, Gen=512, Total=4096 tokens):

| Batch Size | Wall Time ($s$) | `reported_tok_s` | Total Throughput (tok/s) | Gen Goodput (tok/s) | Decode Rate from ITL (tok/s) | TTFT p50 ($ms$) | ITL p50 ($ms$) | p95 E2E ($ms$) | Preempted Seqs | KV Util |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4** | 28.98 | 565.4 | 565.4 | 70.7 | 77.9 | 483.2 | 51.33 | 32,673.3 | 0 | 0.16 |
| **8** | 36.30 | 902.6 | 902.7 | 112.8 | 128.5 | 519.0 | 62.26 | 39,982.9 | 0 | 0.31 |
| **16** | 49.97 | 1311.4 | 1311.5 | 163.9 | 207.3 | 498.3 | 77.20 | 54,602.1 | 0 | 0.62 |
| **24** | 61.16 | **1607.4** | **1607.3** | **200.9** | **249.8** | 500.5 | 96.07 | 69,221.3 | **0** | **0.93** |
| **32** | 94.71 | **1384.0** | **1383.9** | **173.0** | **314.4** | 636.9 | 101.79 | 97,465.7 | **7** | **0.97** |
| **48** | 151.41 | **1298.5** | **1298.5** | **162.3** | **480.0** | 955.4 | 100.00 | 105,427.5 | **23** | **0.97** |

* **Directly Observed Facts:** Throughput peaks at batch 24 ($1607.4\text{ tok/s}$) with 0 preemptions. At batch 32, KV cache saturates at 0.97, 7 sequences are preempted, and throughput drops. At batch 48, 23 sequences are preempted, TTFT nearly doubles to 955.4 ms, p95 latency exceeds 105 seconds, and throughput collapses to 1298.5 tok/s.
* **Serving Mechanism Interpretation:** Preemption is directly observed in the log data. When concurrent sequence memory demand exceeds available KV cache capacity, the scheduler evicts blocks. Re-prefill / recomputation provides a plausible serving mechanism consistent with the observed rise in median TTFT and wall-clock duration.

---

### B3 — Audit of the Misread Column & Goodput Analysis
* **The Misread Column:** `reported_tok_s` ($= \frac{N \times (P+G)}{W}$). It includes parallel prefill tokens ($P=3584$).
* **Batch 24 Goodput:**
  * **Exact Wall-Clock Goodput:** $\frac{24 \times 512\text{ tokens}}{61.16\text{ s}} = \mathbf{200.92\text{ tok/s}}$.
  * **Decode-Phase Rate from ITL:** $\frac{24}{0.09607\text{ s}} = \mathbf{249.82\text{ decode tok/s}}$.
  * *Distinction:* Decode rate measures instantaneous decode iterations; end-to-end goodput includes prefill time and runtime tail overhead. The dataset does not provide separate prefill/decode duration counters for a second exact independent E2E derivation.

---

### B4 — Production Serving Metric
* **Recommended Metric:** **Sequence Preemption Count / Preemption Rate** (`vllm:num_preemptions_total`).
* **Interpretation:** Under safe concurrency ($\le 24$ at 4096 context), preemptions remain **0**. Any preemption indicates that the serving scheduler could not keep all affected sequences resident under the current KV-cache/resource constraints and should be investigated alongside KV utilization and latency.

---

## Part C — Strategic Decision Analysis

### 1. Assignment Facts vs. Stated Assumptions

| Classification | Parameter | Value |
|---|---|---|
| **FACT** | Target Languages | 6 languages: Hindi, Kannada, Tamil, Telugu, Bengali, Marathi |
| **FACT** | Compute Budget | 1× NVIDIA A100-80GB GPU for 2 weeks |
| **FACT** | Reviewer Capacity | 1 native reviewer covering Hindi & Kannada only, 10 h/week |
| **FACT** | Timeline | Launch review in 3 weeks |
| **FACT** | API Budget | $0 external API budget |
| **DERIVED** | Total Available Reviewer Budget | $10\text{ h/week} \times 3\text{ weeks} = \mathbf{30\text{ reviewer-hours}}$ |
| **DERIVED** | Total Available Compute Window | $2\text{ weeks} \times 7\text{ days} \times 24\text{ hours} = \mathbf{336\text{ GPU-hours}}$ |
| **DERIVED** | Native Reviewer Coverage | Direct review covers 2 of 6 languages (Hindi/Kannada); 4 languages lack native review |
| **ASSUMPTION** | Human Review Speed | 2.0 minutes per reviewed pair |
| **ASSUMPTION** | Illustrative SFT Scenario | 1,000 synthetic pairs ($\rightarrow 33.3\text{ reviewer-hours}$) |
| **ASSUMPTION** | Day-1 Test Set | 60 prompts ($30\text{ Hindi} + 30\text{ Kannada}$) across 3 iterations ($\rightarrow 6.0\text{ reviewer-hours}$) |
| **ASSUMPTION** | Proposed Target / Kill Bars | Target: $\ge 70\%$ casual preference; Kill bar: $< 50\%$ preference on Day 7 |

---

### 2. Option Comparison & Recommendation

* **RECOMMENDATION: Option C (Prompt Engineering Only)** on Day 1.
* **Why Option C:**
  1. *Immediate Testability:* Generates empirical feedback on Day 1 without committing GPU training upfront.
  2. *Zero Compute & Deployment Overhead:* Introduces no secondary model weights or pipeline latency.
  3. *Reviewer Feasibility:* Consumes only $\sim 6.0$ of the 30 available reviewer-hours, leaving 24 hours for final release audits.
  4. *Evidence Before Commitment:* Preserves all 336 A100 GPU-hours if the Day-7 Kill Criterion triggers a pivot to SFT.
* **Primary Success Metric:** Casual-tone preference win-rate on blind paired evaluation with factual retention as guardrail.
* **Proposed Kill Criterion:** If by Day 7 Option C achieves $< 50\%$ casual preference win-rate or $< 90\%$ factual accuracy, immediately pivot to Option A (SFT) for the remaining 14 days on the A100.
* **Key Limitation:** The reviewer directly covers Hindi and Kannada only; Tamil, Telugu, Bengali, and Marathi lack direct native validation.

---

## Final Conclusions

1. **Tokenizer Selection Governs Expansion:** Claims that Indic scripts are inherently 6× more expensive were artifacts of testing GPT-2. On modern tokenizers, expansion is 1.26×–2.53×.
2. **Whitespace Fertility Is Flawed for Dravidian Languages:** Whitespace word denominators penalize agglutinative morphology; Relative Token Expansion on parallel text must be used instead.
3. **Serving Capacity Bounded at Batch 24:** At 4096 context on 1× L4, concurrency above batch 24 exceeds KV cache memory, triggering severe preemption and latency inflation.
4. **Prompt Engineering Is the Optimal Day-1 Strategy:** Minimizes deployment complexity and respects the 30-hour reviewer constraint.

---

## Audit Trail & Reproducibility

### Executable Scripts (Relative Paths)
* `partA/scripts/prepare_corpus.py`: Downloads and canonicalizes FLORES-200 parallel corpus.
* `partA/audit/run_audit.py`: Multi-hypothesis forensic audit runner.
* `partA/corrected_analysis/analyze_metrics.py`: Comprehensive cross-tokenizer metric analysis.
* `partB/analysis/capacity_analysis.py`: Mathematical KV cache hardware modeling.
* `partB/analysis/benchmark_analysis.py`: Forensic audit of `bench_log.csv`.
* `partC/analysis_scratch.py`: Arithmetic verification of reviewer and compute allocations.

### Key Data & Result Artifacts
* `partA/corpus/`: Cleaned 997 parallel sentences (`eng.txt`, `hin.txt`, `kan.txt`, `tel.txt`).
* `partA/audit/results/`: Raw experiment outputs (`exp_A_whitespace.json`, `exp_B_lowercasing.json`, `exp_C_aggregation.json`, `exp_E_tokenizers.json`, `exp_norm_unicode.json`).
* `partA/corrected_analysis/results/corrected_metrics.json`: Cross-tokenizer metrics across 4 languages.
* `partB/analysis/results/`: Capacity and benchmark forensic logs (`kv_capacity_derivation.json`, `benchmark_forensics.json`).
* `partA/recommendation_memo.md`: 1-page executive routing memo.
* `partC/memo.md`: Strategic AI decision memo.
