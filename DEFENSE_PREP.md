# FlamAI Technical Defense Preparation Guide

Concise, interview-ready answers for the 30-minute technical defense of "The Audit".

---

### 1. Why this corpus? (FLORES-200 dev split)
* **Answer:** It provides 997 professionally translated, sentence-aligned parallel texts across English (`eng_Latn`), Hindi (`hin_Deva`), Kannada (`kan_Knda`), and Telugu (`tel_Telu`). This strictly controls for semantic content, ensuring cross-lingual token count differences reflect tokenizer encoding efficiency rather than differing prompt lengths or topics.

### 2. What can it NOT tell us?
* **Answer:** It cannot measure live conversational production distributions, user code-switching (Hinglish/Kanglish), long multi-turn context dynamics, or domain-specific jargon (e.g. medical/legal terminology).

### 3. What is the actual `fertility.py` bug?
* **Answer:** `line.split(" ")` at line 62. On multiple consecutive spaces, it produces empty string elements `""`, artificially inflating the word count denominator and deflating calculated fertility by 1.4%–2.0% on typical text. Using Python's default `line.split()` correctly handles arbitrary whitespace.

### 4. Why is lowercasing a measurement distortion rather than universally a bug?
* **Answer:** Devanagari and Dravidian scripts have no uppercase/lowercase distinction. Lowercasing changes English tokenization and reduces the measured English token count by 3.58% in this corpus, while having negligible effect on Hindi. It is a measurement distortion when benchmarking against case-preserved production text.

### 5. Why is macro-averaging not mathematically wrong?
* **Answer:** Macro-average ($\frac{1}{N}\sum \frac{T_i}{W_i}$) measures the expected fertility of an individual sentence. Aggregate micro-average ($\frac{\sum T_i}{\sum W_i}$) measures overall system token volume per word. Neither is mathematically wrong, but micro-average is the appropriate estimand for total infrastructure cost and capacity planning. On sentence data, they diverge by $<1.3\%$.

### 6. Which suspicious-looking code is actually correct?
* **Answer:** `unicodedata.normalize("NFC", line)` at line 49. In Indic scripts, decomposed NFD forms separate vowels into distinct combining marks that fall back to raw byte tokens (+4.15% inflation in Kannada). NFC canonical composition is standard, necessary NLP preprocessing.

### 7. Why these tokenizers?
* **Answer:** To test vocabulary scale and architecture:
  * `GPT-2` (50k vocab, Byte-level BPE): 2019 baseline used in `REPORT_v0.md`.
  * `Meta-Llama-3-8B` (128k vocab, Tiktoken BPE): Matches the 128k vocabulary size specified for `FLM-4B-Instruct`.
  * `Qwen2.5-7B` (152k vocab, BPE): High-capacity multilingual BPE.
  * `XLM-RoBERTa-base` (250k vocab, Unigram SentencePiece): Dedicated multilingual model with balanced Indic coverage.

### 8. Why these denominators?
* **Answer:**
  * **Tokens / Whitespace Word:** Standard NLP metric, but severely distorts agglutinative languages (+35.8% for Kannada).
  * **Tokens / Unicode Character:** Measures code points, but ignores that Indic matras are separate scalar values.
  * **Tokens / Grapheme Cluster:** Measures tokenization density per Unicode extended grapheme cluster.
  * **Tokens / UTF-8 Byte:** Useful secondary diagnostic for byte-level BPE compression efficiency relative to raw text representation.
  * **Tokens / Parallel Sentence (Relative Token Expansion):** Content-controlled primary metric for true serving cost.

### 9. Why tokens per parallel sentence? What is a grapheme?
* **Answer:**
  * *Tokens / Parallel Sentence:* For an aligned parallel corpus, tokens per parallel sentence is the most directly interpretable denominator because the same underlying semantic information is held constant across all languages.
  * *Grapheme:* A grapheme cluster here means a Unicode extended grapheme cluster produced by Unicode grapheme-cluster segmentation (`regex.findall(r"\X", text)`). It is not synonymous with an orthographic syllable or with an Indic akshara in every case.

### 10. Recalculate KV bytes per token.
* **Answer:**
  $$\text{KV bytes/token} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_fp16}$$
  $$= 2 \times 28 \times 8 \times 128 \times 2 = \mathbf{114,688\text{ bytes/token}} = \mathbf{112.0\text{ KiB/token}}$$

### 11. Recalculate 4096-token KV memory.
* **Answer:**
  $$4096 \times 114,688\text{ bytes} = \mathbf{469,762,048\text{ bytes}} = \mathbf{448.0\text{ MiB}} \approx \mathbf{0.4698\text{ GB}}$$

### 12. Recalculate theoretical capacity.
* **Answer:** On a 24 GB GPU at 0.92 utilization ($22.08\text{ GB}$ usable), subtracting 8.40 GB model weights (4.2B fp16) and 1.60 GB overhead leaves **$12.08\text{ GB}$** for KV cache.
  $$\text{Primary Theoretical Capacity} = \frac{12.08 \times 10^9\text{ bytes}}{469,762,048\text{ bytes}} = \mathbf{25.72\text{ sequences (decimal model)}}$$
  *Binary sensitivity case:* Interpreting nominal 24 GB as 24 GiB yields $\approx \mathbf{28.93\text{ sequences}}$.

### 13. Explain empirical implied capacity.
* **Answer:** At batch 24, logged `kv_cache_util` is 0.93 with 0 preemptions. The empirical implied capacity is $\frac{24}{0.93} \approx \mathbf{25.81\text{ sequences}}$, reconciling the modeled KV utilization with the logged 0.93 value at batch 24.

### 14. Explain B2 throughput collapse.
* **Answer:** The CSV directly shows KV pressure, preemptions, higher TTFT, and worse throughput. Re-prefill/recomputation is a plausible mechanism consistent with the observed degradation, but the supplied telemetry does not directly measure the recomputed-token count.

### 15. What is the misread `REPORT_v0.md` column?
* **Answer:** `reported_tok_s`. It measures total tokens processed per second ($\frac{N \times (P+G)}{W}$), combining parallel prefill with serial decode. The intern mistook this for generation serving throughput and falsely claimed longer prompts improve throughput.

### 16. Derive B3 goodput TWO ways.
* **Answer (Batch 24, Prompt 3584, Gen 512, Wall 61.16s, Reported 1607.4 tok/s):**
  * **Method 1 (Direct Output Volume):**
    $$\text{Goodput} = \frac{24 \times 512}{61.16\text{ s}} = \mathbf{200.916\text{ tok/s}} \approx \mathbf{200.92\text{ tok/s}}$$
  * **Method 2 (Reported Throughput × Generation Fraction):**
    $$\text{Goodput} = 1607.4 \times \left(\frac{512}{3584 + 512}\right) = 1607.4 \times \left(\frac{512}{4096}\right) = \mathbf{200.925\text{ tok/s}} \approx \mathbf{200.93\text{ tok/s}}$$
  *(The $0.01\text{ tok/s}$ difference is solely due to the 1-decimal rounding in the logged 1607.4 value).*

### 17. Why is 249.82 not end-to-end goodput?
* **Answer:** $249.82\text{ tok/s}$ ($\frac{24}{0.09607\text{ s}}$) is the **median decode-phase rate estimate** derived from ITL. It only measures instantaneous token emission during the active decode loop, ignoring the ~500ms prefill phase and engine dispatch overhead.

### 18. What single metric confirms B2?
* **Answer:** **`num_preemptions_total`** (e.g. `vllm:num_preemptions_total`). It directly verifies when concurrent active sequences exceed physical KV cache capacity, causing scheduler evictions.

### 19. Why Path C (Prompt Engineering)?
* **Answer:**
  1. *Immediate Testability:* Deploys on Day 1 without committing GPU training time upfront.
  2. *Reviewer Feasibility:* Consumes only $\sim 6.0$ of the 30 available reviewer-hours ($20\%$), preserving bandwidth for release audits.
  3. *Zero Infrastructure Overhead:* Introduces no secondary model weights or pipeline latency.
  4. *Reversible:* Preserves A100 compute for SFT if the Day-7 kill criterion triggers a pivot.

### 20. What is the success threshold?
* **Answer:** A three-way decision framework:
  * **SHIP:** Casual preference $\ge \mathbf{70\%}$ AND factual retention $\ge \mathbf{95\%}$ on blind paired evaluation by the native reviewer.
  * **PIVOT:** Casual preference $< \mathbf{50\%}$ OR factual retention $< \mathbf{90\%}$.
  * **CONTINUE / ITERATE:** Any intermediate result between those boundaries.

### 21. What is the kill criterion?
* **Answer:** If Option C fails the Day-7 criterion (preference $< 50\%$ OR factual retention $< 90\%$), begin the Option A feasibility/pilot work using the remaining available A100 allocation, subject to confirming that the remaining compute window is still available.

### 22. What did AI get wrong during this investigation?
* **Answer:**
  1. Claimed FLORES-200 dev split had 842 articles (corrected to actual 281 articles in metadata).
  2. Claimed download was streaming HTTP (corrected to in-memory buffered fetch).
  3. Used sensationalist wording ("fatal flaw", "conclusively disproves") instead of evidence-calibrated terms.
  4. Presented a pseudo-independent second goodput derivation using ITL before correcting to the generation fraction formula.
  5. Introduced unsupported workforce assumptions (\$30/hr, 5k pairs) which were stripped to strictly adhere to assignment constraints.
