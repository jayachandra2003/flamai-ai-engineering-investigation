# AI Usage Report

## 1. Tools Used

* **ChatGPT (OpenAI):** Used for interactive planning, prompt design, critical review, and methodology evaluation.
* **Antigravity (Google DeepMind):** Used as the local agentic coding environment for script authoring, terminal command execution, data extraction, and documentation formatting.

---

## 2. Division of Work

### Role of ChatGPT
* Structured the investigation plan into distinct forensic phases (Part A, Part B, Part C).
* Formulated candidate hypotheses and audit questions for inspecting `fertility.py` and `bench_log.csv`.
* Challenged uncalibrated claims, absolute phrasing ("fatal", "proves", "guarantees"), and unsupported assumptions.
* Defined evidence-quality criteria for cross-lingual metric selection and capacity modeling.

### Role of Antigravity
* Executed Python commands and automated test scripts within the local workspace environment.
* Created reproducible data processing pipelines (`prepare_corpus.py`, `run_audit.py`, `analyze_metrics.py`, `capacity_analysis.py`, `benchmark_analysis.py`).
* Extracted and formatted raw experimental data into structured JSON logs and Markdown tables.
* Validated filesystem paths, links, and code integrity.

### Role of Human Engineer
* Directed the scope of each investigation stage and set quality standards.
* Reviewed all intermediate model outputs, code logic, and mathematical derivations.
* Rejected unsupported assertions and mandated evidence-calibrated revisions.
* Made the final engineering decisions regarding metric selection (Part A), capacity interpretation (Part B), and the Day-1 operational strategy (Part C).

---

## 3. Important Corrections During the Investigation

During the course of the investigation, human-in-the-loop oversight identified and corrected several preliminary AI-generated statements:

1. **Source Article Count Correction (842 → 281):** Corrected an initial claim that the evaluation corpus drew from 842 articles down to the exact **281 distinct article URLs** present in the FLORES-200 `dev` split.
2. **Download Mechanism Wording:** Corrected descriptions from "streaming HTTP GET" to an in-memory buffered fetch (`requests.get(...)` into `io.BytesIO`).
3. **Calibrated Tokenizer Mismatch Wording:** Replaced overly dramatic language ("fatal flaw", "conclusively disproves") with precise, evidence-based descriptions ("major methodological flaw: tokenizer mismatch") and clarified that GPT-2 tokenization is an established implementation that simply does not represent the 128k FLM-4B production specification.
4. **UTF-8 Byte Metric Neutrality:** Corrected the secondary diagnostic metric description to avoid implying that UTF-8 byte density is linguistically neutral across writing systems.
5. **Reconciliation of `kv_cache_util` (0.93):** Reconciled the theoretical decimal VRAM memory budget ($12.08\text{ GB}$) with the engine's allocated block pool, proving why 24 sequences of 4096 tokens produces exactly $93.3\%$ utilization.
6. **Goodput Derivation Correction:** Refactored the B3 goodput analysis into two algebraically distinct derivations: Method 1 via direct output generation volume over wall-clock duration ($\frac{N \times G}{W} = 200.92\text{ tok/s}$) and Method 2 via total reported throughput scaled by generated-token fraction ($1607.4 \times \frac{512}{4096} = 200.93\text{ tok/s}$), while separating out the median decode-phase rate estimate ($249.82\text{ tok/s}$ from ITL) as an instantaneous decode iteration metric.
7. **Removal of Unsupported Part C Assumptions:** Stripped arbitrary assumptions (e.g. \$30/hr labor, 12 pairs/hr, 5,000 SFT pairs) from being presented as constraints, explicitly grounding Part C in the assignment facts (6 languages, 1 A100 for 2 weeks, 1 reviewer for 10 h/week on Hindi/Kannada, 3-week timeline, $0 API budget).

---

## 4. Verification Statement

All AI-generated scripts, calculations, and analytical conclusions were experimentally executed and verified against raw data files (`bench_log.csv`, `model_spec.md`, FLORES-200 parallel text). AI assistance was used as an interactive accelerator; all final conclusions and audit findings were reviewed and validated by the human engineer.
