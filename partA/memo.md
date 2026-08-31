# Engineering Recommendation Memo: Multilingual Routing & Token Cost Strategy

**To:** FlamAI AI Leadership & Infrastructure Committee  
**From:** AI Engineering Audit Team  
**Date:** August 31, 2026  
**Subject:** Audit of Multilingual Tokenizer Assumptions & Production Routing Metric Selection  

---

### 1. Corrected Headline Numbers

* **FACT (Tokenizer Dependence):** The initial `REPORT_v0.md` finding that Hindi is inherently 6×–7× more expensive than English was based solely on the legacy English-centric `gpt2` tokenizer (50k vocabulary). On our controlled 997-sentence FLORES-200 parallel benchmark:
  * **GPT-2 (50k vocab):** Hindi requires **7.45×** total tokens relative to English ($191,828$ vs. $25,741$ tokens).
  * **LLaMA-3 (128k vocab, matching FLM-4B scale):** Hindi token expansion drops to **2.53×** ($65,361$ tokens).
  * **XLM-RoBERTa (250k vocab):** Hindi token expansion drops to **1.26×** ($36,634$ tokens).
* **FACT (Morphological Distortion in Whitespace Fertility):** Whitespace-word fertility ($\text{tokens} / \text{word}$) distorts cross-language cost comparisons for agglutinative Dravidian languages. On XLM-RoBERTa, Kannada exhibits a fertility ratio of **1.85×** English ($2.57$ vs. $1.38\text{ tok/word}$), yet its actual token footprint for the same parallel content is only **1.37×** English ($39,602$ vs. $28,995$ tokens) because Kannada expresses equivalent semantics in 36% fewer whitespace-delimited words ($15,430$ vs. $20,954$).

---

### 2. Primary Routing & Cost Metric Recommendation

* **RECOMMENDATION:** Adopt the **Relative Token Expansion Ratio on Parallel Evaluation Data** ($\frac{\sum T_{\text{lang}}}{\sum T_{\text{eng}}}$) as the primary offline metric for multilingual routing and token-cost modeling.
* **Justification:** Total token count is a direct input to token-based pricing and a primary driver of sequence-length-dependent prefill compute and KV-cache memory allocation. Evaluating token ratios on parallel content eliminates artificial penalties caused by language-specific word compounding and agglutination, providing a normalized, content-controlled baseline. Offline token counts establish sequence length multipliers, though live latency also depends on runtime batching and hardware execution.

---

### 3. Key Caveats & Limitations

* **CAVEAT (Production Tokenizer Identification):** The proprietary `FLM-4B-Instruct` tokenizer was not directly benchmarked because its tokenizer artifacts were not supplied in the starter kit. The measured ratios from LLaMA-3 (128k) and XLM-RoBERTa (250k) demonstrate vocabulary scaling behavior but do not represent exact FLM-4B production values.
* **CAVEAT (Serving Runtime Dynamics):** Token sequence expansion affects prompt prefill and generation decoding differently. End-to-end request latency, time-to-first-token (TTFT), and GPU memory saturation are governed by the prompt-to-generation token ratio, concurrent batch size, GQA attention efficiency, and L4 memory bandwidth (300 GB/s).

---

### 4. Production Metric to Monitor

* **RECOMMENDED LIVE METRIC:** **Mean Total Tokens per Request by Language relative to English** ($\frac{\bar{T}_{\text{lang, req}}}{\bar{T}_{\text{eng, req}}}$).
* **What It Measures:** The actual average token payload (prompt tokens + generated tokens) processed per user request for each language in live production traffic.
* **Why It Matters:** Validates whether real-world customer prompts and completion lengths align with offline parallel evaluation projections or if language-specific verbosity and code-mixing alter token consumption.
* **Diagnostic Threshold / Trigger:** A proposed alerting threshold could be set by comparing live request ratios with the offline benchmark; the threshold should be calibrated from production traffic rather than assumed from this offline study.

---

### 5. Executive Conclusion

> **RECOMMENDATION:** Base multilingual routing and token cost models on **Relative Token Expansion on parallel reference corpora** rather than whitespace-word fertility. Do not budget a 6× serving cost multiplier or separate Indic routing infrastructure based on GPT-2 benchmarks. Validate the actual `FLM-4B-Instruct` tokenizer directly and monitor **Mean Total Tokens per Request by Language** in production telemetry before enforcing hard language-based routing tiers.
