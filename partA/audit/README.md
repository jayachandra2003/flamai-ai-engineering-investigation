# Part A2: Forensic Audit of `fertility.py` & `REPORT_v0.md`

## 1. Executive Summary of Audit Findings

| Category / Hypothesis | Code Location | Pre-Audit Assumption / Baseline | Post-Audit Empirical Finding | Status | Impact / Evidence-Based Assessment |
|---|---|---|---|:---:|---|
| Category / Hypothesis | Code Location | Pre-Audit Assumption / Baseline | Post-Audit Empirical Finding | Status | Impact / Evidence-Based Assessment |
|---|---|---|---|:---:|---|
| **Methodology / Report Interpretation (Tokenizer Mismatch)** | `REPORT_v0.md:8`, `fertility.py:79` | Tested only GPT-2 (50k vocab, English-centric) and concluded Hindi is inherently 6× more expensive for any tokenizer. | `fertility.py` defaults to GPT-2. The unsupported leap occurred in `REPORT_v0.md` when interpreting GPT-2 evidence as a production-general finding for FLM-4B (128k vocab). On FLORES-200, Hindi expansion ranges from 7.45× (GPT-2) down to 2.53× (LLaMA-3 128k) and 1.26× (XLM-R 250k). | **CONFIRMED** | **METHODOLOGY / INTERPRETATION ISSUE.** Token expansion is strongly tokenizer-dependent. Public tokenizers demonstrate vocabulary scaling but are not direct measurements of proprietary FLM-4B. |
| **Conceptual Metric Distortion (Dravidian Agglutination)** | `fertility.py:64`, `REPORT_v0.md:10` | Assumed `tokens / whitespace_words` is an adequate proxy for inference cost. | Dravidian languages (Kannada, Telugu) are agglutinative (~15.5 words/sent vs 21.0 for English). On XLM-R, Kannada fertility is 1.85× English ($2.57 / 1.38$), while its total token count for the same parallel text is 1.37× English ($39,602 / 28,995$). | **CONFIRMED** | **Conceptual Metric Flaw.** Whitespace-word fertility distorts cross-lingual comparisons between analytic and agglutinative languages. |
| **Whitespace Splitting Implementation** | `fertility.py:62` | `line.split(" ")` creates empty strings on multiple spaces. | Under sample corpora containing double spaces, deflated fertility by 1.4% to 2.0%. On clean FLORES-200, effect is minor (+0.0% to +0.04% for eng/hin, +3.6% for kan). | **CONFIRMED** | **Implementation Flaw.** Exists in code, but magnitude on cleaned text is minor. |
| **Lowercasing Preprocessing** | `fertility.py:60` | `line.lower()` applied to avoid casing noise. | Has negligible (0.01%) effect on Devanagari/Kannada/Telugu, while changing English tokenization and reducing measured English token count by 3.58% in this corpus. | **CONFIRMED** | **Implementation Asymmetry.** Distorts English token counts slightly while having negligible effect on Indic scripts. |
| **Macro vs Micro Aggregation** | `fertility.py:67` | Arithmetic mean of per-line ratios $\frac{1}{N}\sum \frac{T_i}{W_i}$ vs global ratio $\frac{\sum T_i}{\sum W_i}$. | Difference between macro- and micro-average across 997 sentences is $<1.3\%$ across all evaluated languages. | **CONFIRMED** | **Minor Methodological Difference.** Statistically minor on sentence-level corpora. |
| **Suspicious-but-Correct Item (Unicode NFC)** | `fertility.py:49` | `unicodedata.normalize("NFC", line)` alters raw text representation. | Without NFC normalization (e.g. if text is in NFD decomposed format), Indic vowel matras decompose into separate code points, inflating byte-level token counts by up to +4.15% (Kannada). | **CONFIRMED CORRECT** | **Appropriate Preprocessing.** NFC canonical composition is standard practice for Indic text tokenization. |

---

## 2. Detailed Audit of Confirmed Issues

### Issue 1: Tokenizer Selection & Interpretation Mismatch
* **Location:** `REPORT_v0.md:8` (and `fertility.py:79` default setting)
* **Classification:** Methodology / Report Interpretation Issue (not a code defect in `fertility.py`, which accurately executes GPT-2 tokenization).
* **Exact Command:** `python partA/audit/run_audit.py`
* **Before Value (GPT-2 on 997 FLORES-200 parallel sentences):**
  * English: 25,741 tokens
  * Hindi: 191,828 tokens (**7.45× vs. Eng**; fertility 7.80 vs. 1.23)
  * Kannada: 349,772 tokens (**13.59× vs. Eng**; fertility 22.67 vs. 1.23)
  * Telugu: 335,642 tokens (**13.04× vs. Eng**; fertility 20.48 vs. 1.23)
* **After Value (Representative 128k LLaMA-3 Tokenizer on same corpus):**
  * English: 25,792 tokens
  * Hindi: 65,361 tokens (**2.53× vs. Eng**; fertility 2.66 vs. 1.23)
  * Kannada: 229,014 tokens (**8.88× vs. Eng**; fertility 14.84 vs. 1.23)
  * Telugu: 215,433 tokens (**8.35× vs. Eng**; fertility 13.15 vs. 1.23)
* **Absolute Delta (Hindi Tokens):** $191,828 - 65,361 = \mathbf{-126,467\text{ tokens}}$
* **Relative Delta / Token Reduction:** $\frac{-126467}{191828} = \mathbf{-65.93\%}$
* **Direction of Distortion:** Massively inflated apparent Indic token requirements under the legacy 50k GPT-2 tokenizer.
* **Why It Matters:** The intern's report treated GPT-2 results as an inherent linguistic property of Hindi, claiming a universal 6×–7× cost. In reality, expanding vocabulary to 128k reduces Hindi expansion to 2.53×. Public tokenizers illustrate vocabulary scaling but do not directly measure the proprietary FLM-4B tokenizer.

---

### Issue 2: Conceptual Metric Flaw in `tokens / whitespace_words`
* **Location:** `fertility.py:64`, `REPORT_v0.md:10`
* **Classification:** Conceptual Metric Flaw (misleading denominator choice).
* **Exact Command:** `python partA/corrected_fertility.py`
* **Before Value (XLM-RoBERTa Whitespace Fertility Ratio vs. English):**
  * English: $1.3837\text{ tok/word}$ ($28,995\text{ tokens} / 20,954\text{ words}$)
  * Kannada: $2.5666\text{ tok/word}$ ($39,602\text{ tokens} / 15,430\text{ words}$)
  * Apparent Fertility Ratio: $\frac{2.5666}{1.3837} = \mathbf{1.8548\times}$ (+85.5% vs. Eng)
* **After Value (XLM-RoBERTa Total Parallel Token Expansion Ratio vs. English):**
  * Actual Total Token Ratio: $\frac{39,602}{28,995} = \mathbf{1.3658\times}$ (+36.6% vs. Eng)
* **Absolute Delta (Ratio Inflation):** $1.8548 - 1.3658 = \mathbf{+0.4890}$
* **Relative Delta:** $\frac{1.8548 - 1.3658}{1.3658} = \mathbf{+35.80\%}$
* **Direction of Distortion:** Artificially penalizes agglutinative languages where multiple morphemes fuse into fewer whitespace-separated words.
* **Why It Matters:** Kannada expresses the same semantic content in 26.4% fewer whitespace words ($15,430$ vs. $20,954$). Dividing by words creates an artificial +35.8% penalty unrelated to actual sequence length or inference memory demand.

---

### Issue 3: Whitespace Splitting Implementation (`line.split(" ")` vs `line.split()`)
* **Location:** `fertility.py:62`
* **Classification:** Code Implementation Bug.
* **Exact Command:** `python partA/audit/run_audit.py`
* **Before Value (`split(" ")` on starter samples):**
  * English sample: `1.2652 tok/word`
  * Hindi sample: `7.4485 tok/word`
* **After Value (`split()` on starter samples):**
  * English sample: `1.2831 tok/word`
  * Hindi sample: `7.5985 tok/word`
* **Absolute Delta:** English = $+0.0179$, Hindi = $+0.1500$
* **Relative Delta:** English = $\mathbf{+1.41\%}$, Hindi = $\mathbf{+2.01\%}$
* **Direction of Distortion:** Deflated fertility on texts containing multiple consecutive spaces by creating empty string `""` pseudo-words.
* **Why It Matters:** Unsanitized text with varying whitespace generates inaccurate word counts. While the effect on cleaned parallel corpora is small (+0.04% on FLORES-200 Hindi), `str.split()` is standard Python for whitespace tokenization.

---

### Issue 4: Lowercasing Transformation Asymmetry
* **Location:** `fertility.py:60`
* **Classification:** Preprocessing Implementation Asymmetry.
* **Exact Command:** `python partA/audit/run_audit.py`
* **Before Value (Cased text under GPT-2 on 997 FLORES-200 sentences):**
  * English: 25,741 tokens
  * Hindi: 191,828 tokens
* **After Value (Lowercased text `line.lower()` under GPT-2):**
  * English: 26,696 tokens
  * Hindi: 191,842 tokens
* **Absolute Delta:** English = $\mathbf{+955\text{ tokens}}$ (+3.71% when lowercasing), Hindi = $\mathbf{+14\text{ tokens}}$ (+0.01%)
* **Relative Delta:** Lowercasing reduces measured cased English token count by $\mathbf{-3.58\%}$ relative to lowercased text ($\frac{25741 - 26696}{26696} = -3.58\%$).
* **Direction of Distortion:** Changes English tokenization by breaking uppercase acronyms while having negligible effect on Indic scripts.
* **Why It Matters:** Lowercasing introduces a one-sided distortion in cross-lingual comparisons against non-cased scripts like Devanagari, Kannada, and Telugu.

---

### Issue 5: Macro-Average vs. Micro-Average Aggregation
* **Location:** `fertility.py:67`
* **Classification:** Methodological Aggregation Difference.
* **Exact Command:** `python partA/audit/run_audit.py`
* **Before Value (Macro-Average $\frac{1}{N}\sum \frac{T_i}{W_i}$ on LLaMA-3, 997 FLORES-200 sentences):**
  * English: `1.2395` | Hindi: `2.6667` | Kannada: `15.0359` | Telugu: `13.2431`
* **After Value (Micro-Average $\frac{\sum T_i}{\sum W_i}$ on LLaMA-3):**
  * English: `1.2309` | Hindi: `2.6562` | Kannada: `14.8421` | Telugu: `13.1458`
* **Absolute Delta:** English = $-0.0086$, Hindi = $-0.0105$, Kannada = $-0.1938$, Telugu = $-0.0973$
* **Relative Delta:** English = $\mathbf{-0.70\%}$, Hindi = $\mathbf{-0.39\%}$, Kannada = $\mathbf{-1.29\%}$, Telugu = $\mathbf{-0.73\%}$
* **Direction of Distortion:** Macro-averaging slightly overweights short sentences with atypical token/word ratios.
* **Why It Matters:** Macro- and micro-averaging estimate different quantities. Micro-averaging (aggregate ratio) is mathematically appropriate for estimating global serving token volume.

---

## 3. Detailed Audit of Suspicious-but-Correct Feature: Unicode NFC Normalization

* **Hypothesis:** `unicodedata.normalize("NFC", line)` alters raw input strings and might distort tokenization benchmarks.
* **Experiment:** Compare tokenization of NFC composed strings against NFD decomposed strings across 997 FLORES-200 sentences using GPT-2.
* **Measurements:**
  * English: 26,696 tokens (NFC) $\rightarrow$ 26,723 tokens (NFD) [**+27 tokens, +0.10%**]
  * Hindi: 191,842 tokens (NFC) $\rightarrow$ 191,846 tokens (NFD) [**+4 tokens, +0.00%**]
  * Telugu: 335,737 tokens (NFC) $\rightarrow$ 337,770 tokens (NFD) [**+2,033 tokens, +0.61%**]
  * **Kannada:** 349,802 tokens (NFC) $\rightarrow$ 364,303 tokens (NFD) [**+14,501 tokens, +4.15% inflation**]
* **Why It Is Correct:** In Indic scripts, vowel diacritics (matras) and consonant conjuncts can be encoded as precomposed characters (NFC) or decomposed sequences of base letter + combining mark (NFD). Byte-level BPE models segment decomposed code points into fallback byte tokens, causing artificial token inflation (+4.15% in Kannada). Canonical NFC normalization is therefore standard and methodologically necessary.

---

## 4. Quantitative Cross-Tokenizer Evidence Table (FLORES-200, 997 Parallel Sentences)

| Tokenizer | Vocab Size | Language | Total Tokens | Tok / Word (`split()`) | Tok / Char | Tok / UTF-8 Byte | Total Tokens vs. Eng | Fertility vs. Eng |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPT-2** (tiktoken) | 50,257 | **eng** | 25,741 | 1.23 | 0.206 | 0.205 | 1.00× | 1.00× |
| | | **hin** | 191,828 | 7.80 | 1.529 | 0.595 | **7.45×** | **6.35×** |
| | | **kan** | 349,772 | 22.67 | 2.655 | 0.979 | **13.59×** | **18.45×** |
| | | **tel** | 335,642 | 20.48 | 2.639 | 0.991 | **13.04×** | **16.67×** |
| **Meta-Llama-3-8B** (HF) | 128,000 | **eng** | 25,792 | 1.23 | 0.206 | 0.206 | 1.00× | 1.00× |
| | | **hin** | 65,361 | 2.66 | 0.521 | 0.203 | **2.53×** | **2.16×** |
| | | **kan** | 229,014 | 14.84 | 1.738 | 0.641 | **8.88×** | **12.06×** |
| | | **tel** | 215,433 | 13.15 | 1.694 | 0.636 | **8.35×** | **10.68×** |
| **Qwen2.5-7B** (HF) | 151,643 | **eng** | 26,255 | 1.25 | 0.210 | 0.210 | 1.00× | 1.00× |
| | | **hin** | 116,701 | 4.74 | 0.930 | 0.362 | **4.45×** | **3.79×** |
| | | **kan** | 182,074 | 11.80 | 1.382 | 0.509 | **6.93×** | **9.42×** |
| | | **tel** | 185,113 | 11.30 | 1.456 | 0.546 | **7.05×** | **9.02×** |
| **XLM-RoBERTa-base** (HF) | 250,002 | **eng** | 28,995 | 1.38 | 0.232 | 0.231 | 1.00× | 1.00× |
| | | **hin** | 36,634 | 1.49 | 0.292 | 0.114 | **1.26×** | **1.08×** |
| | | **kan** | 39,602 | 2.57 | 0.301 | 0.111 | **1.37×** | **1.85×** |
| | | **tel** | 38,708 | 2.36 | 0.304 | 0.114 | **1.33×** | **1.71×** |

*Note: In the table above, `Meta-Llama-3-8B`, `Qwen2.5-7B`, and `XLM-RoBERTa-base` are public comparison tokenizers illustrating the impact of vocabulary size and multilingual pretraining. They are not asserted to be identical to the FLM-4B production tokenizer.*

---

## 5. What This Evidence Does NOT Establish

1. **Exact FLM-4B Production Fertility:** This evidence does not establish the exact token expansion of FLM-4B-Instruct unless that specific model's tokenizer artifact is directly benchmarked.
2. **Universal Superiority of One Tokenizer:** A lower token count on FLORES-200 does not establish that a tokenizer is universally superior across other tasks, domains, or downstream generation quality.
3. **Direct Inference Latency or Cost:** Offline token counts do not establish live serving throughput, time-to-first-token, or inter-token latency, which also depend on batching dynamics, attention kernels, and hardware constraints.
4. **Equivalence of Token Expansion and Serving Expense:** A 2.5× token expansion on input prompts does not linearly translate to a 2.5× increase in total request cost if generation lengths are short or if serving is compute-bound during prefill.

---

## 6. How to Reproduce All Results

```powershell
# Run the complete automated forensic audit suite
python partA/audit/run_audit.py
```
