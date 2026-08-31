# Part A3: Corrected Multilingual Metric & Tokenizer Analysis

## 1. Production Specification Summary vs. Comparison Tokenizers

To ensure grounded engineering decisions, we distinguish what is directly established in the FLM-4B serving specification from our offline comparative measurements:

| Property | Production Specification ([`bench/model_spec.md`](bench/model_spec.md)) | Comparative Measurement Information ([`analyze_metrics.py`](partA/corrected_analysis/analyze_metrics.py)) |
|---|---|---|
| **Model** | `FLM-4B-Instruct` (4.2B dense params, 28 layers, GQA 8 KV / 24 Q heads) | N/A (Tokenizer-only offline evaluation) |
| **Vocabulary Size** | **128,000 (128k)** | 50,257 (`gpt2`), 128,000 (`llama3`), 151,643 (`qwen2.5`), 250,002 (`xlm-roberta`) |
| **Context Window** | `max_model_len` = 4096 tokens | N/A (Evaluated per-sentence on FLORES-200) |
| **Serving Hardware** | 1× NVIDIA L4 (24 GB, 300 GB/s bandwidth) | Local execution |
| **Tokenizer Weights** | Proprietary / Not provided in starter-kit | Public HuggingFace & Tiktoken models |

*Status:* `Meta-Llama-3-8B` (128k vocab), `Qwen2.5-7B` (152k vocab), and `XLM-RoBERTa-base` (250k vocab) serve as representative comparison points to evaluate vocabulary scale. They are not claimed to be identical to the proprietary FLM-4B production tokenizer.

---

## 2. Evaluation of Candidate Denominator Metrics

| Candidate Metric | Formula | What It Measures | Advantages & Use Cases | Fundamental Limitations | Suitability for Routing / Cost Decisions |
|---|---|---|---|---|:---:|
| **A. Tokens / Whitespace Word** | $\frac{\text{Tokens}}{\text{Words}_{\text{ws}}}$ | Subword fragmentation per space-delimited string. | Quick diagnostic for English-like analytic languages. | Conflates morphology with tokenization; penalizes agglutinative languages (Kannada, Telugu) where words contain multiple morphemes. | **UNSUITABLE** for cross-lingual cost decisions. |
| **B. Tokens / Unicode Character** | $\frac{\text{Tokens}}{\text{Chars}_{\text{unicode}}}$ | Tokenization density per Unicode code point (scalar value). | Less sensitive to whitespace variations. | Indic scripts use diacritics/matras as separate code points; does not reflect perceived typographical units. | **POOR** (Distorts across script types). |
| **C. Tokens / Grapheme Cluster** | $\frac{\text{Tokens}}{\text{Graphemes}}$ | Tokenization density per user-perceived typographical character (akshara). | Accurately treats consonant + combining matras as a single visual unit. | Does not resolve multi-morpheme lexical density differences across languages. | **STRUCTURAL DIAGNOSTIC**. |
| **D. Tokens / UTF-8 Byte** | $\frac{\text{Tokens}}{\text{Bytes}_{\text{utf8}}}$ | Tokenization compression per raw data byte. | Diagnostic for byte-level BPE compression efficiency relative to encoded text representation. | Indic characters require 3 UTF-8 bytes vs 1 byte for ASCII; byte density is not linguistically neutral. | **SECONDARY DIAGNOSTIC ONLY**. |
| **E. Total Tokens on Parallel Content** | $\sum_{i=1}^N T_{i,\text{lang}}$ | Total sequence length generated for identical semantic text. | Directly reflects token sequence length submitted to LLM prefill/KV cache. | Requires parallel reference text to compute baseline. | **EXCELLENT FOR BENCHMARKING**. |
| **F. Relative Token Expansion vs English** | $\frac{\text{Total Tokens}(\text{lang})}{\text{Total Tokens}(\text{eng})}$ | Normalized multiplicative token inflation for equivalent meaning. | Direct multiplier for token-based API pricing, context consumption, and KV cache sizing. | Assumes English as baseline reference. | **RECOMMENDED PRIMARY METRIC**. |
| **G. Macro vs Micro Aggregation** | Macro: $\frac{1}{N}\sum \frac{T_i}{D_i}$<br>Micro: $\frac{\sum T_i}{\sum D_i}$ | Unweighted line average vs global corpus total ratio. | Micro directly models total system token volume. | Macro overweights short lines with extreme ratios. | **MICRO RECOMMENDED**. |

---

## 3. Corrected Results Table (FLORES-200, 997 Parallel Sentences)

All metrics computed directly from [`partA/corpus/`](partA/corpus/) via [`partA/corrected_analysis/analyze_metrics.py`](partA/corrected_analysis/analyze_metrics.py):

| Tokenizer | Vocab Size | Language | Total Tokens | Tok / Word (`split()`) | Tok / Char | Tok / Grapheme | Tok / UTF-8 Byte | Relative Token Expansion vs. Eng | Fertility Ratio vs. Eng |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPT-2** (tiktoken) | 50,257 | **eng** | 25,741 | 1.23 | 0.206 | 0.206 | 0.205 | 1.00× | 1.00× |
| | | **hin** | 191,828 | 7.80 | 1.529 | 1.813 | 0.595 | **7.45×** | **6.35×** |
| | | **kan** | 349,772 | 22.67 | 2.655 | 3.259 | 0.979 | **13.59×** | **18.45×** |
| | | **tel** | 335,642 | 20.48 | 2.639 | 3.578 | 0.991 | **13.04×** | **16.67×** |
| **Meta-Llama-3-8B** (128k scale) | 128,000 | **eng** | 25,792 | 1.23 | 0.206 | 0.206 | 0.206 | 1.00× | 1.00× |
| | | **hin** | 65,361 | 2.66 | 0.521 | 0.618 | 0.203 | **2.53×** | **2.16×** |
| | | **kan** | 229,014 | 14.84 | 1.738 | 2.134 | 0.641 | **8.88×** | **12.06×** |
| | | **tel** | 215,433 | 13.15 | 1.694 | 2.297 | 0.636 | **8.35×** | **10.68×** |
| **Qwen2.5-7B** (152k scale) | 151,643 | **eng** | 26,255 | 1.25 | 0.210 | 0.210 | 0.210 | 1.00× | 1.00× |
| | | **hin** | 116,701 | 4.74 | 0.930 | 1.103 | 0.362 | **4.45×** | **3.79×** |
| | | **kan** | 182,074 | 11.80 | 1.382 | 1.696 | 0.509 | **6.93×** | **9.42×** |
| | | **tel** | 185,113 | 11.30 | 1.456 | 1.974 | 0.546 | **7.05×** | **9.02×** |
| **XLM-RoBERTa-base** (250k scale) | 250,002 | **eng** | 28,995 | 1.38 | 0.232 | 0.232 | 0.231 | 1.00× | 1.00× |
| | | **hin** | 36,634 | 1.49 | 0.292 | 0.346 | 0.114 | **1.26×** | **1.08×** |
| | | **kan** | 39,602 | 2.57 | 0.301 | 0.369 | 0.111 | **1.37×** | **1.85×** |
| | | **tel** | 38,708 | 2.36 | 0.304 | 0.413 | 0.114 | **1.33×** | **1.71×** |

---

## 4. Mathematical Investigation of the Denominator Question

### Why `tokens / whitespace_word` Distorts Cross-Language Comparisons

Let $T_{\text{lang}}$ be the total token count and $W_{\text{lang}}$ be the total whitespace word count. The fertility ratio of a language relative to English is:

$$\text{Fertility Ratio} = \frac{T_{\text{lang}} / W_{\text{lang}}}{T_{\text{eng}} / W_{\text{eng}}} = \left(\frac{T_{\text{lang}}}{T_{\text{eng}}}\right) \times \left(\frac{W_{\text{eng}}}{W_{\text{lang}}}\right) = \text{Token Expansion Ratio} \times \left(\frac{W_{\text{eng}}}{W_{\text{lang}}}\right)$$

Because word segmentation conventions differ fundamentally across language typologies:
- **Analytic English:** 20,954 words across 997 sentences (21.02 words/sent).
- **Agglutinative Kannada:** 15,430 words across 997 sentences (15.48 words/sent) $\rightarrow \frac{W_{\text{eng}}}{W_{\text{kan}}} = \mathbf{1.358}$ (**+35.8% multiplier**).
- **Agglutinative Telugu:** 16,388 words across 997 sentences (16.44 words/sent) $\rightarrow \frac{W_{\text{eng}}}{W_{\text{tel}}} = \mathbf{1.279}$ (**+27.9% multiplier**).
- **Inflected Hindi:** 24,607 words across 997 sentences (24.68 words/sent) $\rightarrow \frac{W_{\text{eng}}}{W_{\text{hin}}} = \mathbf{0.851}$ (**-14.9% multiplier**).

### Empirical Demonstration of Divergence (FLORES-200):

| Tokenizer | Language | Total Tokens | Token Expansion Ratio vs. Eng | Micro Fertility (Tok/Word) | Fertility Ratio vs. Eng | Systematic Divergence |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Meta-Llama-3-8B** | Hindi | 65,361 | **2.53×** | 2.66 | **2.16×** | **-14.8%** |
| | Kannada | 229,014 | **8.88×** | 14.84 | **12.06×** | **+35.8%** |
| | Telugu | 215,433 | **8.35×** | 13.15 | **10.68×** | **+27.9%** |
| **XLM-RoBERTa** | Hindi | 36,634 | **1.26×** | 1.49 | **1.08×** | **-14.8%** |
| | Kannada | 39,602 | **1.37×** | 2.57 | **1.85×** | **+35.8%** |
| | Telugu | 38,708 | **1.33×** | 2.36 | **1.71×** | **+27.9%** |

*Conclusion:* In Kannada and Telugu, whitespace-word fertility artificially inflates apparent inefficiency by **35.8%** and **27.9%**, because the language expresses complex grammatical relationships in compound words rather than separated prepositions and auxiliary verbs.

---

## 5. Decision on Primary Routing / Cost Metric

### **Primary Routing / Cost Metric: Relative Token Expansion Ratio on Parallel Evaluation Data**
$$\text{Metric}_{\text{primary}} = \frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}} \quad \text{(evaluated on parallel semantic corpora)}$$

* **Why It Is Selected:**
  "Total token count is a direct input to token-based pricing and an important driver of sequence-length-dependent compute and KV-cache usage; actual latency and memory behavior also depend on prefill/decode mix, batching, model architecture, and serving configuration."
  Evaluating the Relative Token Expansion Ratio on parallel semantic corpora removes morphological word-segmentation artifacts while providing a standardized, content-controlled multiplier for context window and token usage modeling.

### **Secondary Diagnostic Metric: Tokens per UTF-8 Byte ($\text{Tok} / \text{Byte}$)**
* "Tokens per UTF-8 byte is a useful secondary diagnostic for tokenizer compression relative to the encoded text representation."

### **Key Serving Caveat:**
* Token expansion ratio is a static text representation metric. In production serving, **end-to-end request cost and latency depend on the proportion of prefill (prompt) vs. decode (generation) tokens**, batching dynamics, KV cache saturation limits, and whether execution is in a memory-bound decode regime or compute-bound prefill regime.

---

## 6. What This Evidence Does NOT Establish

1. **Exact FLM-4B Production Tokenizer Ratios:** Does not establish the exact token count or expansion ratio of the proprietary `FLM-4B` model without directly testing its specific tokenizer artifacts.
2. **Universal Model Quality:** A low token expansion ratio does not establish superior downstream generation, reasoning, or translation accuracy.
3. **Live Latency / Throughput from Token Counts Alone:** Offline token counts cannot predict live server latency (TTFT, ITL, p95 E2E) without accounting for serving runtime configurations.

---

## 7. Reproduction Command

```powershell
python partA/corrected_analysis/analyze_metrics.py
```
