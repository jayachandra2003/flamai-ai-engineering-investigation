# Executive Recommendation Memo: Tone Shift Strategy

## Recommendation & Decision
**Deploy Option C (Prompt Engineering)** on Day 1 as the primary strategy for the 3-week launch window. Retain **Option A (Synthetic SFT)** as a secondary escalation path if Prompt Engineering fails its Day-7 kill criterion.

---

## Assignment Facts & Stated Assumptions
* **Facts:**
  * **Goal:** Casual/conversational tone across 6 languages (Hindi, Kannada, Tamil, Telugu, Bengali, Marathi).
  * **Compute Budget:** 1× NVIDIA A100-80GB GPU for 2 weeks ($336\text{ GPU-hours}$).
  * **Reviewer Capacity:** 1 native reviewer covering Hindi and Kannada only, 10 h/week ($30\text{ reviewer-hours}$ total).
  * **Timeline & Budget:** Launch review in 3 weeks; $0 external API budget.
* **Stated Assumptions (Decision Model):**
  * Reviewer speed: 2.0 min/pair ($30\text{ pairs/h}$).
  * Illustrative SFT review load: 1,000 synthetic pairs = $33.3\text{ h}$ (exceeds total budget).
  * Day-1 test suite: 60 prompts ($30\text{ Hindi} + 30\text{ Kannada}$) across 3 iterations ($180\text{ judgements} = 6.0\text{ h}$).

---

## Option Comparison & Arithmetic
| Strategy | Compute Required | Reviewer Hours | Timeline Risk | Core Bottleneck |
|---|:---:|:---:|:---:|---|
| **A. Synthetic SFT** | $\le 336\text{ GPU-h}$ | $33.3\text{ h}$ (1k pairs) | High | Reviewer bottleneck; data bugs leave no pivot time |
| **B. $\le 1\text{B}$ Rewriter** | $\le 336\text{ GPU-h}$ | High | High | Multi-model orchestration and serving latency overhead |
| **C. Prompt Engineering** | **0 GPU-h** | **$6.0\text{ h}$ ($20\%$)** | **Low** | **Zero training risk; immediate Day-1 empirical results** |

* **Arithmetic:** Reviewing 1,000 synthetic pairs requires $\frac{1000 \times 2.0}{60} = 33.3\text{ hours}$, exceeding the entire 30-hour reviewer budget without evaluating trained models. In contrast, prompt engineering consumes only $6.0\text{ hours}$ ($20\%$ of budget), leaving 24 hours for final validation.

## Success Metric, Thresholds & Kill Criterion
* **Metric:** Blind side-by-side preference win-rate ($P_{\text{casual}}$) on Hindi and Kannada with factual accuracy retention ($P_{\text{correct}}$) as a hard guardrail. For each baseline-vs-candidate pair, convert the reviewer judgment into candidate win, baseline win, or tie. Preference win-rate is candidate wins divided by non-tied comparisons (ties reported separately).
* **Three-Way Decision Framework:**
  * **SHIP:** Casual preference $\ge 70\%$ AND factual retention $\ge 95\%$.
  * **PIVOT (Kill Bar):** Casual preference $< 50\%$ OR factual retention $< 90\%$.
  * **CONTINUE / ITERATE:** Any intermediate result between those boundaries.
* **Kill Criterion Action:** If Option C fails the Day-7 criterion, begin the Option A feasibility/pilot work using the remaining available A100 allocation, subject to confirming that the remaining compute window is still available.

---

## Day-1 Experiment & Decision Logic
* **Setup:** Compare baseline formal prompt vs. conversational system prompt on FLM-4B across 30 Hindi and 30 Kannada prompts (60 pairs).
* **Execution:** Measure blind reviewer preference (candidate win / baseline win / tie) and token count delta ($\Delta G$).
* **Why Not Strongest Alternative (Option A):** SFT incurs upfront data curation and training overhead before validating if prompting suffices. Prompt engineering provides immediate empirical signal while preserving the A100 allocation for an escalation path.
* **Key Limitation:** The native reviewer covers Hindi and Kannada only; Tamil, Telugu, Bengali, and Marathi cannot receive native-speaker validation prior to launch without additional reviewer capacity.

---

## Final Decision
Deploy Option C on Day 1. If Day-7 preference is <50% OR factual retention is <90%, pivot to Option A using the remaining available A100 allocation, subject to confirming that the remaining compute window is still available.
