# Part B: Serving Capacity & Throughput Reconciliation

## 1. B1: Theoretical KV Cache Capacity Modeling & Reconciling with `kv_cache_util`

### Model & Hardware Parameters ([bench/model_spec.md](bench/model_spec.md))
* **Model:** `FLM-4B-Instruct` (4.2B parameters, 28 layers, $d_{\text{model}}=3072$)
* **Attention Configuration:** 24 Query heads, 8 KV heads (Grouped-Query Attention / GQA), head dimension $d_k = 128$
* **Data Types / Precision:** fp16 model weights ($2\text{ bytes/param}$), fp16 KV cache ($2\text{ bytes/element}$)
* **Serving Hardware:** 1× NVIDIA L4 (24 GB nominal VRAM, 300 GB/s peak memory bandwidth, 121 TFLOPS fp16 compute)
* **Serving Constraints:** `max_model_len` = 4096 tokens, `gpu_memory_utilization` = 0.92, non-KV runtime overhead $\approx 1.6\text{ GB}$

---

### Step-by-Step Mathematical Derivation

#### **Step 1: KV Cache Memory per Token**
Each token requires storing a Key vector and a Value vector for every layer across all KV heads:
$$\text{KV bytes per token} = 2 \times \text{layers} \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_element}$$
$$\text{KV bytes per token} = 2 \times 28 \times 8 \times 128 \times 2\text{ bytes} = \mathbf{114,688\text{ bytes/token}} = \mathbf{112.0\text{ KiB/token}} = \mathbf{0.1147\text{ MB/token}}$$

#### **Step 2: KV Cache Memory for One 4096-Token Sequence**
$$\text{KV memory per sequence} = 4096\text{ tokens} \times 114,688\text{ bytes/token} = \mathbf{469,762,048\text{ bytes}} = \mathbf{448.0\text{ MiB}} \approx \mathbf{0.4698\text{ GB}}$$

---

### Understanding the Estimated KV Cache Budget & `kv_cache_util`

#### **Decimal GB vs. Binary GiB Conventions:**
1. **Primary Decimal Budget Modeling ($1\text{ GB} = 10^9\text{ bytes}$):**
   * Usable VRAM Budget: $24.0\text{ GB} \times 0.92 = 22.08\text{ GB}$
   * Model Weights (fp16): $4.2 \times 10^9\text{ params} \times 2\text{ bytes} = 8.40\text{ GB}$
   * Non-KV Runtime Overhead: $1.60\text{ GB}$
   * **Available KV Cache Budget:** $22.08 - 8.40 - 1.60 = \mathbf{12.08\text{ GB}}$ ($12.08 \times 10^9\text{ bytes}$)
   * **Theoretical Max Concurrency:** $\frac{12.08 \times 10^9\text{ bytes}}{469,762,048\text{ bytes}} = \mathbf{25.72\text{ sequences}}$

2. **Binary Sensitivity Case ($1\text{ GiB} = 1024^3\text{ bytes}$):**
   * Usable VRAM Budget: $24.0\text{ GiB} \times 0.92 = 22.08\text{ GiB}$ ($22,609.9\text{ MiB}$)
   * Model Weights (fp16): $8.40\times 10^9\text{ bytes} = 7.823\text{ GiB}$ ($8,011.0\text{ MiB}$)
   * Non-KV Overhead: $1.600\text{ GiB}$ ($1,638.4\text{ MiB}$)
   * **Available KV Cache Budget:** $22.08 - 7.823 - 1.600 = \mathbf{12.657\text{ GiB}}$ ($12,960.5\text{ MiB}$)
   * **Theoretical Max Concurrency:** $\frac{12,960.5\text{ MiB}}{448.0\text{ MiB}} = \mathbf{28.93\text{ sequences}}$
   *(This is a sensitivity calculation; the primary result uses the supplied 24 GB specification).*

---

### Reconciling Theoretical Capacity with Logged `kv_cache_util`

1. **What is the Denominator of `kv_cache_util`?**
   Per [bench/model_spec.md](bench/model_spec.md), `kv_cache_util` measures the *peak KV cache block utilization* during the run. Its denominator is the **total number of KV cache blocks pre-allocated into the GPU memory pool by the serving engine at initialization**.
2. **Empirical Implied Capacity vs. Theoretical Capacity:**
   * At Batch 24, all 24 requests run to the full context of $4096\text{ tokens}$ ($3584\text{ prompt} + 512\text{ gen}$), requiring $24 \times 469,762,048\text{ bytes} = \mathbf{11.274\text{ GB}}$ of active KV cache.
   * Under the decimal KV cache memory budget of $\mathbf{12.08\text{ GB}}$:
     $$\text{Modeled KV Cache Utilization} = \frac{11.274\text{ GB}}{12.080\text{ GB}} = \mathbf{0.9333} \approx \mathbf{0.93}$$
     Reconciled the modeled KV utilization with the logged 0.93 value at batch 24; the small difference is attributable to the rounding/representation of the logged utilization.
   * The **empirical implied capacity** from the logged run is $\frac{24}{0.93} \approx \mathbf{25.81\text{ sequences}}$, which aligns closely with the theoretical decimal upper bound ($25.72\text{ sequences}$).
3. **What the Benchmark Records at Batch 32 and Batch 48:**
   * At Batch 32, memory demand is $32 \times 0.4698\text{ GB} = \mathbf{15.03\text{ GB}}$ ($124.4\%$ of capacity). The block pool saturates at **0.97**, and the benchmark logs **7 preemptions**.
   * At Batch 48, memory demand is $48 \times 0.4698\text{ GB} = \mathbf{22.55\text{ GB}}$ ($186.7\%$ of capacity). The pool saturates at **0.97**, and the benchmark logs **23 preemptions**.
   * *Note on Evidence Discipline:* The benchmark explicitly records 7 and 23 preemptions. The capacity calculation demonstrates that capacity was exceeded, but does not mathematically predict those exact eviction counts.

---

## 2. B2: Benchmark Reconciliation & Throughput Anomaly Analysis

### Full Data from [bench/bench_log.csv](bench/bench_log.csv) (Prompt=3584, Gen=512, Total=4096 tokens)

| Batch Size | Wall Time ($s$) | `reported_tok_s` | Total Tok/s | Gen Goodput (tok/s) | Decode Rate from ITL (tok/s) | TTFT p50 ($ms$) | ITL p50 ($ms$) | p95 E2E ($ms$) | Preempted Seqs | KV Util |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4** | 28.98 | 565.4 | 565.4 | 70.7 | 77.9 | 483.2 | 51.33 | 32,673.3 | 0 | 0.16 |
| **8** | 36.30 | 902.6 | 902.7 | 112.8 | 128.5 | 519.0 | 62.26 | 39,982.9 | 0 | 0.31 |
| **16** | 49.97 | 1311.4 | 1311.5 | 163.9 | 207.3 | 498.3 | 77.20 | 54,602.1 | 0 | 0.62 |
| **24** | 61.16 | **1607.4** | **1607.3** | **200.9** | **249.8** | 500.5 | 96.07 | 69,221.3 | **0** | **0.93** |
| **32** | 94.71 | **1384.0** | **1383.9** | **173.0** | **314.4** | 636.9 | 101.79 | 97,465.7 | **7** | **0.97** |
| **48** | 151.41 | **1298.5** | **1298.5** | **162.3** | **480.0** | 955.4 | 100.00 | 105,427.5 | **23** | **0.97** |

---

### Mechanism of the Anomaly (Observed vs. Inference vs. Prediction)

* **OBSERVED (Directly from CSV):**
  1. Concurrency scaled stably from batch 4 to batch 24 with $0$ preemptions and rising throughput ($565.4 \rightarrow 1607.4\text{ tok/s}$).
  2. At batch 32, `kv_cache_util` saturated at **0.97**, **7 sequences were preempted**, and reported throughput dropped to $1384.0\text{ tok/s}$.
  3. At batch 48, **23 sequences were preempted**, wall-clock time increased to $151.41\text{ s}$, median TTFT nearly doubled ($500.5\text{ ms} \rightarrow 955.4\text{ ms}$), p95 latency reached $105.4\text{ s}$, and reported throughput dropped to $1298.5\text{ tok/s}$.
* **INFERENCE (Serving Mechanism):**
  * Preemption is directly observed. When active sequence memory demand exceeds available KV cache capacity, the scheduler preempts requests and evicts their blocks.
  * In standard paged-attention architectures, re-prefill/recomputation provides a plausible serving mechanism consistent with the observed throughput and latency degradation, as re-processing evicted prompt tokens on resumption matches the observed rise in median TTFT ($500\text{ ms} \rightarrow 955\text{ ms}$) and total wall-clock duration.
* **PREDICTION (Batch Limiting Proposal):**
  * If a serving system limits concurrency to `max-num-seqs = 24`, processing 48 requests in two consecutive batches of 24 is projected as a first-order estimate to take $\approx 2 \times 61.16\text{s} = \mathbf{122.32\text{ seconds}}$, improving upon the measured thrashing duration of $151.41\text{ s}$.

---

## 3. B3: Audit of the Misread Column & Two Independent Goodput Derivations

### The Misread Column: `reported_tok_s`

* **Definition:**
  $$\text{reported\_tok\_s} = \frac{\text{num\_requests} \times (\text{prompt\_len} + \text{gen\_len})}{\text{wall\_clock\_s}} = \text{Total Processed Tokens / Second}$$
  It combines parallel, compute-bound **prefill tokens** ($P=3584$) with memory-bandwidth-bound **generation decode tokens** ($G=512$).
* **How `REPORT_v0.md` Interpreted It:**
  1. Claimed long prompts yield superior throughput ($1311\text{ tok/s}$ at batch 16 long vs $883\text{ tok/s}$ at batch 16 short). In reality, for long prompts, $87.5\%$ of tokens are prompt tokens processed during prefill. Actual generation goodput is **$44.3\%$ lower** for long prompts ($163.9\text{ tok/s}$ vs $294.5\text{ tok/s}$) due to KV memory overhead and higher ITL ($77.20\text{ ms}$ vs $48.33\text{ ms}$).
  2. Linearly extrapolated $1600\text{ tok/s}$ at batch 24 to claim $\sim 3200\text{ tok/s}$ at batch 48, ignoring the physical memory capacity ceiling and the fact that batch 48 actually logged only $1298.5\text{ tok/s}$.

---

### Two Independent Derivations of Batch-24 Goodput ($P=3584, G=512, N=24, W=61.16\text{s}$)

#### **Method 1: Direct Generation Volume over Wall-Clock Duration**
$$\text{Goodput}_{\text{Method 1}} = \frac{N \times G}{W} = \frac{24 \times 512\text{ tokens}}{61.16\text{ seconds}} = \frac{12,288}{61.16} = \mathbf{200.916\text{ tok/s}} \approx \mathbf{200.92\text{ tok/s}}$$

#### **Method 2: Reported Throughput Scaled by Generated-Token Fraction**
$$\text{Goodput}_{\text{Method 2}} = \text{reported\_tok\_s} \times \left(\frac{G}{P + G}\right) = 1607.4 \times \left(\frac{512}{3584 + 512}\right) = 1607.4 \times \left(\frac{512}{4096}\right) = \mathbf{200.925\text{ tok/s}} \approx \mathbf{200.93\text{ tok/s}}$$
*(Explanation: The $0.01\text{ tok/s}$ difference between Method 1 and Method 2 is entirely caused by the 1-decimal rounding in the logged `reported_tok_s` value: $1607.4$ vs. unrounded $1607.325\text{ tok/s}$).*

#### **Separate Metric: Median Decode-Phase Rate Estimate (from ITL)**
During the active generation phase, 24 tokens are emitted across the batch every $\text{ITL} = 96.07\text{ ms} = 0.09607\text{ s}$:
$$\text{Rate}_{\text{decode}} = \frac{\text{Batch Size}}{\text{ITL}_{\text{seconds}}} = \frac{24}{0.09607\text{ s}} = \mathbf{249.82\text{ decode tok/s}}$$
*Distinction:* This measures instantaneous generation speed during the memory-bound decode iterations only, excluding prompt prefill latency ($\text{TTFT} \approx 500\text{ ms}$) and engine overhead. It is not an end-to-end goodput derivation.

---

## 4. B4: Single Recommended Production Validation Metric

* **Recommended Metric:** **`num_preemptions_total`** (e.g. Prometheus / vLLM metric `vllm:num_preemptions_total`).
* **Why It Is Diagnostic:** Directly tests the preemption component of the hypothesis. In the tested configurations up to batch 24, preemptions were 0. Under higher tested concurrency, preemptions increased to 7 at batch 32 and 23 at batch 48.
* **Limitation:** The counter validates the occurrence of scheduler evictions; it does not by itself capture subsequent queue delay or recomputation latency, which should be monitored alongside TTFT and p95 E2E latency.

---

## 5. Reproduction Commands

```powershell
# Run theoretical KV cache capacity derivation
python partB/analysis/capacity_analysis.py

# Run benchmark log forensic analysis
python partB/analysis/benchmark_analysis.py
```
