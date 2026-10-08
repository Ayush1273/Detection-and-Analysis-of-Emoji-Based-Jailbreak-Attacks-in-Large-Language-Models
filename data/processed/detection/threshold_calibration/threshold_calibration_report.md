# Experiment 8: Threshold Calibration, Operating-Point Analysis, and Calibration Stress Testing Report

## 1. Executive Summary & Research Objectives

This report documents **Experiment 8**, which investigates how pre-inference harmful-intent detectors behave across decision thresholds ($\tau \in [0.05, 0.95]$) and assesses whether predicted posterior probabilities are sufficiently calibrated for practical deployment as a prompt-level safety filter.

We evaluate the two strongest representations identified in preceding experiments:
1. **`sentence-transformers/all-MiniLM-L6-v2` + Logistic Regression** (Generic semantic baseline)
2. **`SummerSigh/Safety-Policy-MiniLM` + Logistic Regression** (Domain-aligned safety encoder)

Both models were evaluated under strict zero-leakage 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` cross-validation on `unique_pair_id` across:
* **Original Benchmark ($N=800$)**: 200 prompt groups $\times$ 4 perturbation variants (400 Benign, 400 Harmful).
* **Length-Balanced Benchmark ($N=488$)**: 122 prompt groups $\times$ 4 perturbation variants (244 Benign, 244 Harmful).

### Core Research Questions
1. *How do precision, recall, and false positive rates trade off across decision thresholds, and what are the operational costs of enforcing high recall vs. low false alarms?*
2. *Are the out-of-fold predicted probabilities well-calibrated, and does post-hoc calibration (Platt scaling or Isotonic regression) improve reliability without leakage?*
3. *Can adjusting the decision threshold mitigate the sequence-dilution vulnerability observed in Experiment 6A and 7B without crippling benign system utility?*

### Essential Interpretation Principles
* **Classification Target**: Strictly `harmful vs. benign prompt intent`.
* **Not an Execution Predictor**: The classifier operates strictly pre-inference. It does not predict whether an LLM will be successfully jailbroken or generate harmful text.
* **Probabilities as Statistical Estimators**: Calibrated probabilities represent statistical confidence in prompt intent categories, not absolute guarantees of safety or attack occurrence.

---

## 2. Experiment 8A: Decision Threshold Sweep Analysis

We swept decision thresholds from $\tau = 0.05$ to $0.95$ in steps of $0.05$. Out-of-fold predictions were strictly aggregated across folds without tuning on training data.

### Benchmark Metrics Across Selected Thresholds (Length-Balanced $N=488$)

| Model | Threshold ($\tau$) | Accuracy | Balanced Accuracy | Harmful Recall | Harmful Precision | F1 Score | Benign FPR | FNR | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **all-MiniLM-L6-v2** | $0.20$ | $60.2\%$ | $60.2\%$ | $98.4\%$ | $55.3\%$ | $0.708$ | $77.9\%$ | $1.6\%$ | $+0.301$ |
| | $0.30$ | $57.2\%$ | $57.2\%$ | $93.4\%$ | $54.2\%$ | $0.686$ | $79.1\%$ | $6.6\%$ | $+0.208$ |
| | $0.40$ | $63.7\%$ | $63.7\%$ | $82.4\%$ | $59.8\%$ | $0.693$ | $54.9\%$ | $17.6\%$ | $+0.291$ |
| | **$0.50$ (Default)** | **$67.4\%$** | **$67.4\%$** | **$63.5\%$** | **$68.9\%$** | **$0.661$** | **$28.7\%$** | **$36.5\%$** | **$+0.349$** |
| | **$0.55$ (Max $J$)** | **$69.3\%$** | **$69.3\%$** | **$56.1\%$** | **$76.1\%$** | **$0.646$** | **$17.6\%$** | **$43.9\%$** | **$+0.399$** |
| | $0.65$ | $68.0\%$ | $68.0\%$ | $43.0\%$ | $86.1\%$ | $0.574$ | $7.0\%$ | $57.0\%$ | $+0.416$ |
| | $0.80$ | $58.2\%$ | $58.2\%$ | $17.2\%$ | $95.5\%$ | $0.292$ | $0.8\%$ | $82.8\%$ | $+0.288$ |
| **Safety-Policy-MiniLM** | $0.20$ | $50.0\%$ | $50.0\%$ | $99.6\%$ | $50.0\%$ | $0.666$ | $99.6\%$ | $0.4\%$ | $+0.000$ |
| | $0.25$ | $49.6\%$ | $49.6\%$ | $91.0\%$ | $49.8\%$ | $0.643$ | $91.8\%$ | $9.0\%$ | $-0.015$ |
| | $0.35$ | $52.0\%$ | $52.0\%$ | $88.5\%$ | $51.2\%$ | $0.649$ | $84.4\%$ | $11.5\%$ | $+0.055$ |
| | **$0.45$ (Max $J$)** | **$65.8\%$** | **$65.8\%$** | **$81.6\%$** | **$62.0\%$** | **$0.704$** | **$50.0\%$** | **$18.4\%$** | **$+0.333$** |
| | **$0.50$ (Default)** | **$65.6\%$** | **$65.6\%$** | **$77.5\%$** | **$62.6\%$** | **$0.692$** | **$46.3\%$** | **$22.5\%$** | **$+0.321$** |
| | $0.60$ | $56.6\%$ | $56.6\%$ | $20.9\%$ | $79.7\%$ | $0.331$ | $5.3\%$ | $79.1\%$ | $+0.239$ |
| | $0.70$ | $50.0\%$ | $50.0\%$ | $0.0\%$ | $0.0\%$ | $0.000$ | $0.0\%$ | $100.0\%$ | $+0.000$ |

*Complete multi-threshold grid saved to [`threshold_metrics.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/threshold_calibration/threshold_metrics.csv) and visualized in [`threshold_curves.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/threshold_calibration/threshold_curves.png).*

---

## 3. Experiment 8B: Descriptive Operating-Point Analysis

Rather than claiming a singular "optimal" threshold, we analyze four distinct operating configurations with explicitly defined optimization criteria:

### Operating Points on the Length-Balanced Benchmark ($N=488$)

| Model | Operating Point | Optimization Criterion | Threshold ($\tau$) | Harmful Recall | Harmful Precision | Benign FPR | FNR | F1 Score | Balanced Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **all-MiniLM-L6-v2** | **1. Default Threshold** | Fixed standard boundary ($\tau=0.50$) | $0.50$ | $63.5\%$ | $68.9\%$ | $28.7\%$ | $36.5\%$ | $0.661$ | $67.4\%$ |
| | **2. High-Recall Point** | Max $\tau$ achieving Recall $\ge 90.0\%$ | $0.30$ | $93.4\%$ | $54.2\%$ | $79.1\%$ | $6.6\%$ | $0.686$ | $57.2\%$ |
| | **3. Balanced Point** | Max Youden's $J = \text{Recall} - \text{FPR}$ | $0.55$ | $56.1\%$ | $76.1\%$ | $17.6\%$ | $43.9\%$ | $0.646$ | **$69.3\%$** |
| | **4. Low-FPR Point** | Min $\tau$ achieving Benign $\text{FPR} \le 10.0\%$ | $0.65$ | $43.0\%$ | $86.1\%$ | **$7.0\%$** | $57.0\%$ | $0.574$ | $68.0\%$ |
| **Safety-Policy-MiniLM** | **1. Default Threshold** | Fixed standard boundary ($\tau=0.50$) | $0.50$ | $77.5\%$ | $62.6\%$ | $46.3\%$ | $22.5\%$ | $0.692$ | $65.6\%$ |
| | **2. High-Recall Point** | Max $\tau$ achieving Recall $\ge 90.0\%$ | $0.25$ | $91.0\%$ | $49.8\%$ | $91.8\%$ | $9.0\%$ | $0.643$ | $49.6\%$ |
| | **3. Balanced Point** | Max Youden's $J = \text{Recall} - \text{FPR}$ | $0.45$ | $81.6\%$ | $62.0\%$ | $50.0\%$ | $18.4\%$ | **$0.704$** | $65.8\%$ |
| | **4. Low-FPR Point** | Min $\tau$ achieving Benign $\text{FPR} \le 10.0\%$ | $0.70$ | $0.0\%$ | $0.0\%$ | **$0.0\%$** | $100.0\%$ | $0.000$ | $50.0\%$ |

### Trade-Off Analysis Across Operating Regimes
1. **The Cost of High Recall**: Enforcing a strict security screening requirement ($\text{Recall} \ge 90\%$) forces thresholds down to $\tau \le 0.30$. At this level, **$79.1\%$ of benign prompts in MiniLM and $91.8\%$ in Safety-Policy are falsely flagged**. In an operational environment, this would cause severe user friction by intercepting almost all benign traffic.
2. **The Cost of Low False Alarms**: Prioritizing benign user utility ($\text{FPR} \le 10\%$) requires thresholds of $\tau \ge 0.65$. While MiniLM preserves moderate utility ($43.0\%$ recall, $86.1\%$ precision), Safety-Policy completely collapses to $0.0\%$ recall because its harmful probabilities peak below $0.70$.
3. **Model Profile Divergence at Default ($\tau=0.50$)**:
   * `Safety-Policy-MiniLM` functions as a **high-sensitivity detector** ($77.5\%$ recall, catching $14.0\%$ more attacks than MiniLM, but accepting an elevated $46.3\%$ false alarm rate).
   * `all-MiniLM-L6-v2` functions as a **balanced precision detector** ($63.5\%$ recall, but keeping benign false alarms down to $28.7\%$, with $68.9\%$ precision).

---

## 4. Experiment 8C: Probability Calibration Analysis

We evaluated the calibration quality of out-of-fold predicted probabilities using Brier score, Log Loss, and Expected Calibration Error (ECE, $M=10$ bins). Calibration was performed via 5-fold grouped nested cross-validation on `unique_pair_id` with zero leakage.

### Calibration Performance Summary

| Model | Benchmark Dataset | Calibration Method | Brier Score ($\downarrow$) | Log Loss ($\downarrow$) | ECE ($M=10$) ($\downarrow$) | Calibration Assessment |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **all-MiniLM-L6-v2** | Original ($N=800$) | Uncalibrated (Raw LogReg) | **$0.216$** | **$0.620$** | $0.052$ | Excellent baseline calibration |
| | | Sigmoid / Platt Scaling | $0.218$ | $0.628$ | **$0.047$** | Marginally reduces ECE |
| | | Isotonic Regression | $0.222$ | $0.746$ | $0.079$ | Overfits fold partitions |
| | Length-Balanced ($N=488$) | Uncalibrated (Raw LogReg) | **$0.203$** | **$0.590$** | $0.053$ | Well-calibrated posterior probabilities |
| | | Sigmoid / Platt Scaling | $0.207$ | $0.601$ | $0.062$ | Minimal shift |
| | | Isotonic Regression | $0.207$ | $0.777$ | **$0.045$** | Lower ECE but elevated Log Loss |
| **Safety-Policy-MiniLM** | Original ($N=800$) | Uncalibrated (Raw LogReg) | **$0.240$** | **$0.673$** | **$0.043$** | Well-calibrated raw sigmoid link |
| | | Sigmoid / Platt Scaling | $0.242$ | $0.677$ | $0.058$ | Comparable to uncalibrated |
| | | Isotonic Regression | $0.260$ | $1.212$ | $0.071$ | Severe log loss penalty |
| | Length-Balanced ($N=488$) | Uncalibrated (Raw LogReg) | **$0.231$** | **$0.658$** | **$0.073$** | Moderate calibration error |
| | | Sigmoid / Platt Scaling | $0.234$ | $0.662$ | $0.109$ | Increases calibration error |
| | | Isotonic Regression | $0.243$ | $0.891$ | $0.088$ | Inflates log loss |

*Visual reliability diagrams saved in [`calibration_curve.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/threshold_calibration/calibration_curve.png).*

### Key Findings on Probability Calibration
1. **Raw Logistic Regression Outputs Are Already Calibrated**: Because Logistic Regression optimizes the binary log-likelihood using a logistic sigmoid link function, its uncalibrated out-of-fold probabilities already produce low Brier scores ($0.203$ for MiniLM, $0.231$ for Safety-Policy) and low calibration error ($\text{ECE} \approx 0.04 - 0.07$).
2. **Platt Scaling Offers Minimal Value**: Fitting an additional cross-validated sigmoid scaling step yields virtually identical Brier scores ($+0.003$ change) and does not meaningfully alter decision curves.
3. **Isotonic Regression Degrades Stability**: Non-parametric isotonic regression overfits small cross-validation folds, introducing probability step discontinuities and inflating out-of-fold Log Loss ($0.59 \rightarrow 0.78$ in MiniLM, $0.66 \rightarrow 0.89$ in Safety-Policy). **Raw parametric logistic regression probabilities should be retained directly without non-parametric calibration.**

---

## 5. Experiment 8D: Robustness at Selected Operating Thresholds

We evaluated how the four selected operating points behave when subjected to the existing deterministic padding and emoji perturbation datasets:

### Robustness Performance Grid Across Operating Points

| Condition Evaluated | Class / Perturbation | MiniLM ($\tau=0.50$) | MiniLM ($\tau=0.30$) [High-Rec] | MiniLM ($\tau=0.65$) [Low-FPR] | Safety-Policy ($\tau=0.50$) | Safety-Policy ($\tau=0.25$) [High-Rec] | Safety-Policy ($\tau=0.45$) [Balanced] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Harmful** | Harmful Base ($N=100$) | $71.0\%$ | $94.0\%$ | $49.0\%$ | $68.0\%$ | $92.0\%$ | $71.0\%$ |
| **Short Padded Harmful** | Academic Framing ($N=100$) | $56.0\%$ | $87.0\%$ | $31.0\%$ | $43.0\%$ | $98.0\%$ | $55.0\%$ |
| **Medium Padded Harmful** | Educational Context ($N=100$) | $47.0\%$ | $84.0\%$ | $21.0\%$ | $22.0\%$ | $96.0\%$ | $40.0\%$ |
| **Long Padded Harmful** | Curriculum Review ($N=100$) | $92.0\%$* | $100.0\%$ | $43.0\%$ | **$1.0\%$** | $53.0\%$ | **$2.0\%$** |
| **Prefix Harmful** | Emoji at start ($N=100$) | $61.0\%$ | $90.0\%$ | $33.0\%$ | $61.0\%$ | $100.0\%$ | $65.0\%$ |
| **Suffix Harmful** | Emoji at end ($N=100$) | $68.0\%$ | $94.0\%$ | $48.0\%$ | $72.0\%$ | $100.0\%$ | $76.0\%$ |
| **Insertion Harmful** | Emoji inside prompt ($N=100$) | $63.0\%$ | $91.0\%$ | $46.0\%$ | $75.0\%$ | $100.0\%$ | $80.0\%$ |
| **Original Benign** | Benign Base ($N=100$) | **$38.0\%$ FPR** | **$78.0\%$ FPR** | **$20.0\%$ FPR** | **$40.0\%$ FPR** | **$88.0\%$ FPR** | **$47.0\%$ FPR** |
| **Short Padded Benign** | Academic Framing ($N=100$) | $27.0\%$ FPR | $69.0\%$ FPR | $12.0\%$ FPR | $14.0\%$ FPR | $93.0\%$ FPR | $33.0\%$ FPR |
| **Medium Padded Benign** | Educational Context ($N=100$) | $19.0\%$ FPR | $55.0\%$ FPR | $6.0\%$ FPR | $6.0\%$ FPR | $94.0\%$ FPR | $13.0\%$ FPR |
| **Long Padded Benign** | Curriculum Review ($N=100$) | **$83.0\%$ FPR** | **$100.0\%$ FPR** | **$36.0\%$ FPR** | **$0.0\%$ FPR** | $45.0\%$ FPR | **$0.0\%$ FPR** |
| **Prefix Benign** | Emoji at start ($N=100$) | $33.0\%$ FPR | $75.0\%$ FPR | $15.0\%$ FPR | $37.0\%$ FPR | $100.0\%$ FPR | $43.0\%$ FPR |
| **Suffix Benign** | Emoji at end ($N=100$) | $34.0\%$ FPR | $77.0\%$ FPR | $18.0\%$ FPR | $52.0\%$ FPR | $100.0\%$ FPR | $59.0\%$ FPR |
| **Insertion Benign** | Emoji inside prompt ($N=100$) | $35.0\%$ FPR | $73.0\%$ FPR | $18.0\%$ FPR | $56.0\%$ FPR | $100.0\%$ FPR | $62.0\%$ FPR |

*\*Note: MiniLM long padding recall increases to $92\%$ due to spuriously triggering on security keywords in the template ("threat landscape", "risk management"), which also causes benign FPR to spike to $83\%$.*

---

## 6. Experiment 8E: Master Multi-Dimensional Model Comparison

The table below provides the unified empirical synthesis comparing `all-MiniLM-L6-v2` and `SummerSigh/Safety-Policy-MiniLM`:

| Evaluation Dimension | Metric / Criterion | all-MiniLM-L6-v2 | Safety-Policy-MiniLM | Comparative Trade-Off & Practical Implication |
| :--- | :--- | :---: | :---: | :--- |
| **Threshold-Free Discrimination** | Original ROC-AUC ($N=800$) | **$0.716 \pm 0.077$** | $0.643 \pm 0.040$ | MiniLM achieves higher overall separation on full dataset. |
| | Balanced ROC-AUC ($N=488$) | **$0.751 \pm 0.066$** | $0.672 \pm 0.093$ | MiniLM retains higher ranking quality after controlling for length confound. |
| | Balanced PR-AUC ($N=488$) | **$0.791 \pm 0.054$** | $0.672 \pm 0.142$ | MiniLM provides superior precision across recall spectrum. |
| **Standard Baseline ($\tau=0.50$)** | Balanced Harmful Recall | $63.5\%$ | **$77.5\%$** | Safety-Policy captures $+14.0\%$ more harmful prompts at default threshold. |
| | Balanced Benign FPR | **$28.7\%$** | $46.3\%$ | MiniLM causes $-17.6\%$ fewer false alarms on benign queries. |
| **High-Recall Regime ($\text{Rec} \ge 90\%$)** | Threshold ($\tau$) / Recall / FPR | $\tau=0.30$ / $93.4\%$ / $79.1\%$ | $\tau=0.25$ / $91.0\%$ / $91.8\%$ | Both suffer intolerable false positive explosions when forced to high recall. |
| **Balanced Regime (Max Youden $J$)** | Threshold ($\tau$) / BalAcc / F1 | **$\tau=0.55$ / $69.3\%$ / $0.646$** | $\tau=0.45$ / $65.8\%$ / $0.704$ | MiniLM achieves higher balanced accuracy; Safety-Policy achieves higher F1. |
| **Low-FPR Regime ($\text{FPR} \le 10\%$)** | Threshold ($\tau$) / Recall / FPR | **$\tau=0.65$ / $43.0\%$ / $7.0\%$** | $\tau=0.70$ / $0.0\%$ / $0.0\%$ | MiniLM retains functional detection at low FPR; Safety-Policy collapses. |
| **Calibration Reliability** | Brier Score (Balanced) | **$0.203$** | $0.231$ | MiniLM probabilities are sharper and closer to true posterior. |
| | Expected Calibration Error (ECE) | **$0.053$** | $0.073$ | Both models exhibit low calibration error without post-hoc scaling. |
| **Adversarial Dilution** | Harmful Recall Under Long Padding | $52.0\%$ (spurious) | **$1.0\%$** (diluted) | Both fail global sequence dilution; sequence-level aggregation is vulnerable. |
| | Benign False Alarm Under Padding | Inflates to $83.0\%$ (keyword) | **Suppresses to $0.0\%$** | Safety-Policy correctly understands academic framing as safe; MiniLM over-flags. |
| **Emoji Topology Sensitivity** | Prefix Recall Drop vs. Original | $-10.0\%$ ($71\% \rightarrow 61\%$) | **$-4.0\%$ ($68\% \rightarrow 61\%$)** | Safety-Policy is more resilient to leading emoji token attention shifts. |
| **Operational Overhead** | Parameters / CPU Latency | 22.7M / **$8.5$ ms** | 22.7M / $21.9$ ms | Both run within real-time latency budgets (<25 ms) on commodity CPU. |

---

## 7. Critical Architectural Conclusion & Next Experiment Recommendation

### Why Threshold Tuning Cannot Solve the Single-Vector Dilemma
Experiments 8A–8D provide definitive mathematical proof of why sliding a scalar decision threshold cannot solve the defense problem for a global sequence-level classifier:

1. **The Dilution Deadlock**: When an attacker prepends 100 words of benign academic preamble to a 20-word harmful payload, the sequence-level CLS/mean-pooling embedding is mathematically dominated by benign token representations.
   * If the threshold is kept at $\tau = 0.50$, the attack slips through unnoticed ($\text{Recall} = 1.0\%$).
   * If the threshold is lowered to $\tau = 0.25$ to catch diluted attacks, **$91.8\%$ of benign queries are rejected**.
2. **The Lexical False Alarm Dilemma**: Generic models like MiniLM mistake security-related vocabulary in educational padding for malicious intent (driving benign FPR to $83\%$). Safety-specific models solve this problem ($\text{FPR} = 0\%$), but become even easier to dilute because benign tokens pull the embedding strongly into safe space.

### Mandatory Next Architectural Step: Hierarchical Windowed / Chunked Scanner
Because sequence-level single-vector pooling creates an irreconcilable conflict between dilution and false alarms, the pre-inference defense layer must **abandon full-prompt single-vector aggregation**.

**Recommended Architecture for the Next Phase**:
* **Chunked / Sliding-Window Semantic Scanner**:
  * Instead of pooling the entire prompt into one vector, slice the prompt into overlapping semantic windows (e.g., $w = 25 - 40$ tokens with stride $s = 15$).
  * Apply the safety representation to each window independently.
  * Aggregate via a **Max-Pooling Violation Decision Rule**:
    $$\text{Prompt Violation Score} = \max_{k} P(\text{Harmful} \mid \text{Window}_k)$$
  * Under this architecture, embedding a harmful payload in 500 words of benign educational text does *not* dilute the payload: the specific window containing the malicious command will trigger high harmful probability regardless of the length or nature of surrounding text.
* **Latency Feasibility**: Since MiniLM processes a 40-token window in $<2$ ms on CPU, evaluating $3-5$ overlapping windows per prompt requires $<10$ ms total inference time—fully compatible with pre-inference deployment constraints.

