# Recommendation

## Decision

**RECOMMENDATION: Option C (Prompt Engineering Only)** as the primary Day-1 strategy for the 3-week launch window, with **Option A (Synthetic SFT)** held as a secondary escalation path if prompt engineering fails its Day-7 kill criterion.

---

## Constraints

### Assignment Facts
* **Product Objective:** Make replies casual/conversational across 6 languages: **Hindi, Kannada, Tamil, Telugu, Bengali, Marathi** (current outputs are too formal/textbook).
* **Compute Budget:** **1× NVIDIA A100-80GB GPU for 2 weeks** ($\mathbf{336\text{ GPU-hours}}$ available).
* **Human Reviewer Capacity:** **1 native-speaker reviewer covering Hindi and Kannada only, available 10 h/week** ($\mathbf{30\text{ reviewer-hours}}$ total before launch).
* **Project Timeline:** **Launch review in 3 weeks**.
* **API Budget:** **$0 external API budget** (all generation/evaluation must run locally).

### Stated Assumptions *(Decision Assumptions)*
* **Reviewer Speed:** Assumed at **2.0 minutes per reviewed pair** ($\sim 30\text{ pairs/hour}$).
* **Illustrative SFT Scenario:** **1,000 synthetic pairs** ($500\text{ Hindi} + 500\text{ Kannada}$) used solely to model reviewer throughput.
* **Day-1 Test Set:** **60 prompts** ($30\text{ Hindi} + 30\text{ Kannada}$) across 3 prompt iterations ($180\text{ judgements total}$).
* **Target & Kill Bars:** Proposed decision thresholds ($\ge 70\%$ preference target; $< 50\%$ preference Day-7 kill bar).

---

## Option Comparison

| Option | Training Compute | Reviewer Workload | Deployment Complexity | 3-Week Timeline Risk |
|---|---|---|---|---|
| **A. Synthetic SFT** | Local fine-tuning ($\le 336\text{ GPU-h}$) | **High:** Curating synthetic pairs competes with evaluating model outputs | High (model retraining & regression risk) | High (data bugs leave no pivot time) |
| **B. $\le 1\text{B}$ Rewriter** | Local training ($\le 336\text{ GPU-h}$) | **High:** Requires dual-model dataset curation & validation | High (adds multi-model pipeline latency) | High (complex pipeline orchestration) |
| **C. Prompt Engineering** | **Zero** | **Low:** $\sim 6.0\text{ hours}$ to test Hindi/Kannada prompt variants | **Minimal** (standard single-model inference) | **Low** (Day-1 deployment; results by Day 7) |

---

## Key Arithmetic

1. **Reviewer Capacity:**
   $$\text{Total Available Reviewer Time} = 10\text{ h/week} \times 3\text{ weeks} = \mathbf{30\text{ reviewer-hours}}$$
2. **SFT Reviewer Bottleneck:**
   Under the illustrative assumption of 1,000 synthetic pairs and 2 minutes of review per pair, full review would require **$33.3\text{ hours}$**, exceeding the 30 reviewer-hours available before launch without evaluating any trained checkpoints.
3. **Prompt Engineering Feasibility:**
   Evaluating 60 test prompts across 3 iterative prompt variants ($\sim 180\text{ paired judgements}$) requires $\frac{180 \times 2.0}{60} = \mathbf{6.0\text{ reviewer-hours}}$ ($20\%$ of available budget), preserving 24 reviewer-hours for final release audits.
4. **Compute Allocation:**
   $2\text{ weeks} \times 7\text{ days} \times 24\text{ hours} = \mathbf{336\text{ GPU-hours}}$ available on the A100. Prompt engineering leaves this compute available for offline batch evaluation or immediate SFT pivoting.

---

## Success Metric + Threshold

* **Primary Metric:** **Casual-Tone Preference Win Rate ($P_{\text{casual}}$)** on blind, side-by-side evaluations (Baseline Formal Prompt vs. Treatment Conversational Prompt) scored by the native reviewer on Hindi and Kannada, with **Factual Correctness Retention ($P_{\text{correct}}$)** as a strict guardrail.
* **Proposed Target Threshold:** *(Decision assumption)*
  * **Tone:** $\ge \mathbf{70\%}$ win-rate preference for casual tone over baseline.
  * **Accuracy Guardrail:** $\ge \mathbf{95\%}$ factual retention relative to baseline.

---

## Kill Criterion

> **KILL CRITERION (Proposed Decision Threshold):** If Option C fails the Day-7 criterion, begin the Option A feasibility/pilot work using the remaining available A100 allocation, subject to confirming that the remaining compute window is still available.

---

## Day-1 Experiment

* **Setup:** Compare baseline formal prompt vs. conversational system prompt on FLM-4B across 30 **Hindi** and 30 **Kannada** prompts (60 prompts total).
* **Measurements:**
  1. *Primary:* Blind native-reviewer scoring on casual tone preference (1–3 scale) and factual retention.
  2. *Secondary Diagnostic:* Generated token count ($\Delta G$) to monitor output verbosity.
* **Action:** If Day-1 results show positive tone shift ($\ge 50\%$) with preserved accuracy, continue prompt refinement through Week 1; if results show no tone movement, trigger the SFT pivot early.

---

## Why Not the Strongest Alternative?

* **Strongest Alternative: Option A (Synthetic SFT)**.
* **Rejection Rationale:** SFT introduces upfront training risk and data curation overhead before establishing whether prompt engineering can achieve the desired tone. Starting with prompt engineering provides empirical evidence on Day 1 while preserving available A100 compute if an SFT pivot is required on Day 7.

---

## Key Limitation

* **Six-Language Reviewer Limitation:** The available native-speaker reviewer directly covers Hindi and Kannada only. Therefore the Day-1 experiment provides direct native-language evidence for those two languages, not equivalent native-language validation for Tamil, Telugu, Bengali, and Marathi. Deploying all 6 languages carries unvalidated linguistic risk until native reviewers are onboarded.

---

## Final Decision

> **Deploy Option C (Prompt Engineering)** on Day 1. Focus the native reviewer on evaluating Hindi and Kannada conversational prompt variants. Enforce the **Day-7 Kill Criterion** to pivot to synthetic SFT if prompt engineering fails the 50% preference bar.
