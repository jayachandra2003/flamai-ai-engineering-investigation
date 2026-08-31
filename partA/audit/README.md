# Part A2: Forensic Audit of `fertility.py` & `REPORT_v0.md`

## 1. Executive Summary of Audit Findings

| Category / Hypothesis | Code Location | Pre-Audit Assumption / Baseline | Post-Audit Empirical Finding | Status | Impact / Evidence-Based Assessment |
|---|---|---|---|:---:|---|
| **Tokenizer Model Mismatch** | `fertility.py:79`, `REPORT_v0.md:8` | Tested only GPT-2 (50k vocab, English-centric 2019 model) and concluded Hindi is inherently 6× more expensive for any tokenizer. | GPT-2 tokenization is an established implementation, but using it to make production FLM-4B routing/cost claims is unsupported because `model_spec.md` specifies a 128k vocabulary. Across benchmarked tokenizers on FLORES-200, Hindi token expansion ranges from 7.45× (GPT-2) down to 2.53× (LLaMA-3 128k) and 1.26× (XLM-R 250k). | **CONFIRMED** | **MAJOR METHODOLOGICAL FLAW: tokenizer mismatch.** The observed expansion is strongly tokenizer-dependent and cannot be attributed solely to the script. |
| **Conceptual Metric Distortion (Dravidian Agglutination)** | `fertility.py:64`, `REPORT_v0.md:10` | Assumed `tokens / whitespace_words` is an adequate proxy for inference cost. | Dravidian languages (Kannada, Telugu) are agglutinative (~15.5 words/sent vs 21.0 for English). On XLM-R, Kannada fertility is 1.85× English ($2.57 / 1.38$), while its total token count for the same parallel text is 1.37× English ($39,602 / 28,995$). | **CONFIRMED** | **Conceptual Metric Flaw.** Whitespace-word fertility distorts cross-lingual comparisons between analytic and agglutinative languages. |
| **Whitespace Splitting Implementation** | `fertility.py:62` | `line.split(" ")` creates empty strings on multiple spaces. | Under sample corpora containing double spaces, deflated fertility by 1.4% to 2.0%. On clean FLORES-200, effect is minor (+0.0% to +0.04% for eng/hin, +3.6% for kan). | **CONFIRMED** | **Implementation Flaw.** Exists in code, but magnitude on cleaned text is minor. |
| **Lowercasing Preprocessing** | `fertility.py:60` | `line.lower()` applied to avoid casing noise. | Has 0.0% effect on Devanagari/Kannada/Telugu (no casing in scripts), but inflates English GPT-2 token counts by 3.58% (and LLaMA-3 by 1.97%) by fragmenting uppercase tokens (e.g. `NASA`, `GPU`). | **CONFIRMED** | **Implementation Asymmetry.** Distorts English token counts slightly while having zero effect on Indic scripts. |
| **Macro vs Micro Aggregation** | `fertility.py:67` | Arithmetic mean of per-line ratios $\frac{1}{N}\sum \frac{T_i}{W_i}$ vs global ratio $\frac{\sum T_i}{\sum W_i}$. | Difference between macro- and micro-average across 997 sentences is $<1.3\%$ across all evaluated languages. | **CONFIRMED** | **Minor Methodological Difference.** Statistically minor on sentence-level corpora. |
| **Suspicious-but-Correct Item (Unicode NFC)** | `fertility.py:49` | `unicodedata.normalize("NFC", line)` alters raw text representation. | Without NFC normalization (e.g. if text is in NFD decomposed format), Indic vowel matras decompose into separate code points, inflating byte-level token counts by up to +4.15% (Kannada). | **CONFIRMED CORRECT** | **Appropriate Preprocessing.** NFC canonical composition is standard practice for Indic text tokenization. |

---

## 2. Detailed Audit of Confirmed Issues

### Issue 1: Tokenizer Selection Mismatch
* **Hypothesis:** Benchmarking GPT-2 (50k English-centric vocabulary) does not reflect modern multilingual tokenizers or the 128k vocabulary size specified for FLM-4B in [`bench/model_spec.md`](bench/model_spec.md).
* **Experiment:** Tokenize the 997 parallel sentences of [`partA/corpus/`](partA/corpus/) across four distinct tokenizers:
  1. `gpt2` (50,257 vocab, baseline)
  2. `NousResearch/Meta-Llama-3-8B` (128,000 vocab, representative 128k BPE)
  3. `Qwen/Qwen2.5-7B` (151,643 vocab, modern multilingual BPE)
  4. `xlm-roberta-base` (250,002 vocab, multilingual SentencePiece)
* **Before Values (GPT-2 on FLORES-200):**
  * English: 25,741 tokens (fertility: 1.23 tok/word, 0.206 tok/char)
  * Hindi: 191,828 tokens (fertility: 7.80 tok/word, 1.529 tok/char) $\rightarrow$ **7.45× tokens vs. English**
  * Kannada: 349,772 tokens (fertility: 22.67 tok/word, 2.655 tok/char) $\rightarrow$ **13.59× tokens vs. English**
  * Telugu: 335,642 tokens (fertility: 20.48 tok/word, 2.639 tok/char) $\rightarrow$ **13.04× tokens vs. English**
* **After Values (Representative Comparison Tokenizers on FLORES-200):**
  * **LLaMA-3 (128k):** Hindi total tokens = 65,361 (**2.53× vs. Eng**; fertility 2.66 vs. 1.23)
  * **Qwen 2.5 (152k):** Hindi total tokens = 116,701 (**4.45× vs. Eng**; fertility 4.74 vs. 1.25)
  * **XLM-RoBERTa (250k):** Hindi total tokens = 36,634 (**1.26× vs. Eng**; fertility 1.49 vs. 1.38)
* **Percentage Change / Token Reduction relative to GPT-2:**
  * Hindi token count drops from 191,828 (GPT-2) to 65,361 (LLaMA-3, **-65.9%**) and 36,634 (XLM-R, **-80.9%**).
* **Conclusion:** The claim in `REPORT_v0.md` that Hindi fertility is an immutable property of the script is not supported. Hindi token expansion is strongly tokenizer-dependent.
* **Limitations:** These comparison tokenizers illustrate vocabulary scaling effects but do not measure the proprietary FLM-4B tokenizer directly unless its specific tokenizer files are evaluated.

---

### Issue 2: Conceptual Metric Flaw in `tokens / whitespace_words`
* **Hypothesis:** Dividing token counts by whitespace-separated words creates an artificial penalty for agglutinative languages (e.g. Kannada, Telugu) relative to analytic languages (English).
* **Experiment:** Compare whitespace word fertility ($\frac{\text{Tokens}}{\text{Words}}$) against the total token expansion ratio ($\frac{\text{Tokens}_{\text{lang}}}{\text{Tokens}_{\text{eng}}}$) on the 997 parallel FLORES-200 sentences.
* **Measurements (XLM-RoBERTa on FLORES-200):**
  * English: 20,954 words $\rightarrow$ 28,995 tokens (Fertility = 1.38)
  * Kannada: 15,430 words $\rightarrow$ 39,602 tokens (Fertility = 2.57)
* **Apparent Fertility Ratio:** $\frac{2.57}{1.38} = \mathbf{1.85\times}$ (appears 85% higher).
* **Actual Total Token Ratio for Same Content:** $\frac{39,602}{28,995} = \mathbf{1.37\times}$ (only 37% higher).
* **Conclusion:** Token count is a more direct proxy for token-based inference payload than tokens-per-whitespace-word. Actual serving cost and latency also depend on model architecture, batching, hardware, and serving runtime behavior.
* **Limitations:** Total token count on parallel text controls for content, but production traffic mixes languages and domain distributions differently than benchmark corpora.

---

### Issue 3: Whitespace Splitting Implementation (`line.split(" ")` vs `line.split()`)
* **Hypothesis:** `line.split(" ")` treats consecutive spaces as empty word tokens `""`, artificially deflating fertility.
* **Experiment:** Compare `line.split(" ")` against `line.split()` across starter samples and FLORES-200.
* **Starter Corpus Measurements:**
  * English: `1.2652` $\rightarrow$ `1.2831` (+0.0179, **+1.41%**)
  * Hindi: `7.4485` $\rightarrow$ `7.5985` (+0.1500, **+2.01%**)
* **FLORES-200 Measurements:**
  * English: `1.2825` $\rightarrow$ `1.2826` (+0.00%)
  * Hindi: `7.8232` $\rightarrow$ `7.8260` (+0.04%)
  * Kannada: `22.1483` $\rightarrow$ `22.9456` (+0.7973, **+3.60%**)
  * Telugu: `20.3995` $\rightarrow$ `20.6243` (+0.2249, **+1.10%**)
* **Conclusion:** `line.split(" ")` is an implementation flaw that alters word counts when multiple spaces exist. On sanitized corpora, its aggregate impact is minor.
* **Limitations:** Only affects lines with consecutive whitespace characters.

---

### Issue 4: Lowercasing Transformation Asymmetry
* **Hypothesis:** Lowercasing before tokenization modifies English token counts by breaking uppercase subwords while having zero effect on native Indic scripts.
* **Experiment:** Measure token counts and fertility with `line.lower()` versus original case across 997 FLORES-200 sentences.
* **Measurements (GPT-2):**
  * English: 26,696 tokens (Lower) $\rightarrow$ 25,741 tokens (Cased) [**-955 tokens, -3.58%**]
  * Hindi: 191,842 tokens (Lower) $\rightarrow$ 191,828 tokens (Cased) [**-14 tokens, -0.01%**]
  * Kannada: 349,802 tokens (Lower) $\rightarrow$ 349,772 tokens (Cased) [**-30 tokens, -0.01%**]
  * Telugu: 335,737 tokens (Lower) $\rightarrow$ 335,642 tokens (Cased) [**-95 tokens, -0.03%**]
* **Measurements (LLaMA-3):**
  * English: 26,311 tokens (Lower) $\rightarrow$ 25,792 tokens (Cased) [**-519 tokens, -1.97%**]
  * Hindi: 65,363 tokens (Lower) $\rightarrow$ 65,361 tokens (Cased) [**-2 tokens, -0.00%**]
* **Conclusion:** Lowercasing slightly inflates English token counts (by ~2% to ~3.6%) while leaving Indic scripts unaffected, introducing a minor asymmetry into relative fertility comparisons.
* **Limitations:** Effect size on English depends on the frequency of acronyms and proper nouns in the evaluation text.

---

### Issue 5: Macro-Average vs. Micro-Average Aggregation
* **Hypothesis:** Arithmetic mean of line-level ratios ($\frac{1}{N}\sum \frac{T_i}{W_i}$) differs from global corpus ratio ($\frac{\sum T_i}{\sum W_i}$).
* **Measurements (LLaMA-3 on FLORES-200):**
  * English: Macro `1.2395` vs. Micro `1.2309` [**-0.0086, -0.70%**]
  * Hindi: Macro `2.6667` vs. Micro `2.6562` [**-0.0105, -0.39%**]
  * Kannada: Macro `15.0359` vs. Micro `14.8421` [**-0.1938, -1.29%**]
  * Telugu: Macro `13.2431` vs. Micro `13.1458` [**-0.0973, -0.73%**]
* **Conclusion:** The difference between macro- and micro-average is minor ($<1.3\%$) on sentence-length text, though micro-average is mathematically preferable for modeling total token throughput.
* **Limitations:** Macro-averaging can produce larger variance on corpora with high sentence-length dispersion.

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
