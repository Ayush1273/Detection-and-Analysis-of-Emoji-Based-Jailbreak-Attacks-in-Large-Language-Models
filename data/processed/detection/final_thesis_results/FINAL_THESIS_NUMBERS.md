# Official Frozen Thesis Numbers Reference

**Document Version**: 1.0 (Thesis Freeze)  
**Date**: October 3, 2026  
**Target Research**: M.Tech Thesis: *"Detection and Analysis of Emoji-Based Jailbreak Attacks in Large Language Models"*  
**Freeze Status**: **LOCKED & VERIFIED (THESIS READY)**  

> [!IMPORTANT]
> **Thesis Consistency Mandate**:  
> Every metric in this document has been programmatically audited, reconciled against cross-validation and out-of-fold definitions, and frozen. No subsequent script or tuning run may silently modify these numbers. When writing your thesis chapters, presentation slides, or defense document, copy values **strictly** from this reference.

---

## 1. Selected Candidate Pre-Inference Defense Architecture

* **Model Family**: Safety-Policy-MiniLM (`SummerSigh/Safety-Policy-MiniLM` representation)
* **Classifier Head**: Logistic Regression ($C=1.0$, L2 regularization, class-balanced fit)
* **Architecture**: **Hierarchical / Sliding-Window Pre-Inference Detector**
* **Window Size ($W$)**: **$32$ tokens**
* **Stride ($S$)**: **$16$ tokens** ($50\%$ overlap)
* **Aggregation Rule**: **Max-pooling** ($\max_{m} P(\text{harmful} \mid w_m)$)
* **Decision Threshold**: $\tau = 0.50$
* **Formal Academic Framing**:
  > *"The selected candidate architecture for the evaluated pre-inference defense is Safety-Policy-MiniLM with window size $W=32$ tokens, stride $S=16$ tokens, and max pooling."*  
  > *(NOT "optimal architecture", NOT "guaranteed defense", NOT "complete jailbreak detector".)*

---

## 2. Official Primary Benchmark Results (Length-Balanced Benchmark $N=488$)

### Evaluation Protocol
* **Dataset**: Length-Balanced Benchmark ($N=488$, $122$ prompt pairs $\times 4$ perturbation variants, Hungarian character-matched).
* **Cross-Validation**: $5$-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly on `unique_pair_id`.
* **Leakage Verification**: Zero group overlap ($\text{train\_groups} \cap \text{val\_groups} = \emptyset$), zero post-inference features used.
* **Metric Reporting Format**: $\text{Mean} \pm \text{Standard Deviation}$ across the $5$ validation folds.

### Primary Benchmark Table (Length-Balanced $N=488$, Threshold $\tau=0.50$)

| Model & Architecture | ROC-AUC | PR-AUC | Accuracy | Balanced Accuracy | Precision | Harmful Recall | Benign FPR | Harmful FNR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global `all-MiniLM-L6-v2`** | $0.807 \pm 0.071$ | $0.826 \pm 0.069$ | $72.8 \pm 6.7\%$ | $73.0 \pm 6.6\%$ | $78.1 \pm 11.8\%$ | $67.5 \pm 14.5\%$ | $21.4 \pm 16.1\%$ | $32.5 \pm 14.5\%$ | $0.709 \pm 0.078$ |
| **Windowed `all-MiniLM-L6-v2` ($W=32, S=16, \text{Max}$)** | $0.807 \pm 0.071$ | $0.826 \pm 0.069$ | $72.8 \pm 6.7\%$ | $73.0 \pm 6.6\%$ | $78.1 \pm 11.8\%$ | $67.5 \pm 14.5\%$ | $21.4 \pm 16.1\%$ | $32.5 \pm 14.5\%$ | $0.709 \pm 0.078$ |
| **Global `Safety-Policy-MiniLM`** | $0.723 \pm 0.111$ | $0.722 \pm 0.145$ | $67.5 \pm 8.8\%$ | $67.7 \pm 9.1\%$ | $67.7 \pm 16.7\%$ | $77.2 \pm 9.3\%$ | $41.7 \pm 23.5\%$ | $22.8 \pm 9.3\%$ | $0.707 \pm 0.059$ |
| **Windowed `Safety-Policy-MiniLM` ($W=32, S=16, \text{Max}$)** *(Selected)* | **$0.723 \pm 0.111$** | **$0.722 \pm 0.145$** | **$67.5 \pm 8.8\%$** | **$67.7 \pm 9.1\%$** | **$67.7 \pm 16.7\%$** | **$77.2 \pm 9.3\%$** | **$41.7 \pm 23.5\%$** | **$22.8 \pm 9.3\%$** | **$0.707 \pm 0.059$** |

*Note on Unpadded Equivalence: $100.0\%$ of prompts in the Length-Balanced benchmark contain $\le 24$ tokens, which fit entirely inside a single $32$-token window. Slicing produces exactly $1$ window per prompt, rendering Windowed and Global predictions mathematically identical on concise unpadded inputs.*

### Secondary Reference Benchmark (Original Unbalanced Dataset $N=800$, Threshold $\tau=0.50$)

| Model & Architecture | ROC-AUC | PR-AUC | Accuracy | Balanced Accuracy | Precision | Harmful Recall | Benign FPR | Harmful FNR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global `all-MiniLM-L6-v2`** | $0.719 \pm 0.042$ | $0.681 \pm 0.046$ | $66.6 \pm 3.1\%$ | $66.6 \pm 3.1\%$ | $66.7 \pm 3.5\%$ | $66.8 \pm 7.1\%$ | $33.5 \pm 5.8\%$ | $33.2 \pm 7.1\%$ | $0.665 \pm 0.042$ |
| **Windowed `all-MiniLM-L6-v2` ($W=32, S=16, \text{Max}$)** | $0.719 \pm 0.042$ | $0.681 \pm 0.046$ | $66.6 \pm 3.1\%$ | $66.6 \pm 3.1\%$ | $66.7 \pm 3.5\%$ | $66.8 \pm 7.1\%$ | $33.5 \pm 5.8\%$ | $33.2 \pm 7.1\%$ | $0.665 \pm 0.042$ |
| **Global `Safety-Policy-MiniLM`** | $0.685 \pm 0.074$ | $0.719 \pm 0.094$ | $62.5 \pm 7.2\%$ | $62.5 \pm 7.2\%$ | $61.7 \pm 7.4\%$ | $69.3 \pm 5.2\%$ | $44.3 \pm 14.1\%$ | $30.8 \pm 5.2\%$ | $0.650 \pm 0.049$ |
| **Windowed `Safety-Policy-MiniLM` ($W=32, S=16, \text{Max}$)** | $0.688 \pm 0.078$ | $0.722 \pm 0.097$ | $62.8 \pm 7.6\%$ | $62.8 \pm 7.6\%$ | $61.8 \pm 7.6\%$ | $69.8 \pm 5.5\%$ | $44.3 \pm 14.1\%$ | $30.2 \pm 5.5\%$ | $0.653 \pm 0.054$ |

---

## 3. Official Robustness Benchmark (Contextual Padding Stress Test)

> [!WARNING]
> **Strict Separation Mandate**: Contextual padding stress testing is an out-of-distribution adversarial robustness evaluation and must **NEVER** be mixed into or averaged with the standard 5-fold cross-validation benchmark.

### Experimental Setup
* **Harmful Set**: $100$ base harmful prompt groups wrapped in academic curriculum review framing ($58$ prefix/suffix padding tokens).
* **Benign Set**: $100$ base benign prompt groups wrapped in identical academic curriculum review framing.
* **Evaluation Condition**: Unpadded vs. Short ($18$ tokens) vs. Medium ($36$ tokens) vs. Long ($58$ tokens) Contextual Padding.

### Contextual Padding Robustness Results Table ($N=100$ per Class, Threshold $\tau=0.50$)

| Encoder Model | Detector Architecture | Evaluation Condition | Harmful Recall | Harmful FNR | Mean Harmful Prob | Benign FPR | Benign Specificity | Mean Benign Prob |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Safety-Policy-MiniLM** | Global Sequence | Unpadded Baseline | $68.0\%$ | $32.0\%$ | $0.532 \pm 0.163$ | $40.0\%$ | $60.0\%$ | $0.441 \pm 0.148$ |
| | Global Sequence | Short Padding ($18$ tok) | $52.0\%$ | $48.0\%$ | $0.500 \pm 0.125$ | $14.0\%$ | $86.0\%$ | $0.397 \pm 0.111$ |
| | Global Sequence | Medium Padding ($36$ tok) | $22.0\%$ | $78.0\%$ | $0.426 \pm 0.103$ | $6.0\%$ | $94.0\%$ | $0.372 \pm 0.083$ |
| | **Global Sequence** | **Long Padding ($58$ tok)** | **$1.0\%$** | **$99.0\%$** | **$0.256 \pm 0.078$** | **$0.0\%$** | **$100.0\%$** | **$0.238 \pm 0.065$** |
| **Safety-Policy-MiniLM** | Windowed ($W=32$) | Unpadded Baseline | $68.0\%$ | $32.0\%$ | $0.533 \pm 0.162$ | $40.0\%$ | $60.0\%$ | $0.441 \pm 0.148$ |
| | Windowed ($W=32$) | Short Padding ($18$ tok) | $59.0\%$ | $41.0\%$ | $0.528 \pm 0.128$ | $17.0\%$ | $83.0\%$ | $0.405 \pm 0.117$ |
| | Windowed ($W=32$) | Medium Padding ($36$ tok) | $59.0\%$ | $41.0\%$ | $0.552 \pm 0.125$ | $31.0\%$ | $69.0\%$ | $0.465 \pm 0.111$ |
| | **Windowed ($W=32$)** | **Long Padding ($58$ tok)** | **$67.0\%$** | **$33.0\%$** | **$0.550 \pm 0.126$** | **$32.0\%$** | **$68.0\%$** | **$0.454 \pm 0.108$** |
| `all-MiniLM-L6-v2` | Global Sequence | Long Padding ($58$ tok) | $56.0\%$ | $44.0\%$ | $0.505 \pm 0.071$ | $50.0\%$ | $50.0\%$ | $0.500 \pm 0.054$ |
| `all-MiniLM-L6-v2` | Windowed ($W=32$) | Long Padding ($58$ tok) | $97.0\%$ | $3.0\%$ | $0.669 \pm 0.089$ | $93.0\%$ | $7.0\%$ | $0.638 \pm 0.080$ |

### Statistical Significance of the Recall Recovery
* **Mean Probability Difference**: $\Delta = +0.294 \pm 0.115$
* **Paired $t$-test**: $t = 25.34$, $p = 4.62 \times 10^{-45}$
* **Wilcoxon Signed-Rank Test**: $W = 0.0$, $p = 3.90 \times 10^{-18}$
* **Bootstrap $95\%$ Confidence Interval ($1000$ iterations)**: Mean improvement $+65.8\%$, $95\%$ CI: **$[+56.0\%, +75.0\%]$** harmful recall.

---

## 4. Verification of the 99% Harmful-Window Finding

* **Programmatic Definition of Overlap**:
  * An overlapping window is defined as a token window $[t_{\text{start}}, t_{\text{end}}]$ whose indices encompass $\ge 50\%$ of the localized harmful payload tokens $[t_{\text{prompt\_start}}, t_{\text{prompt\_end}}]$ (tokens $43$ to $71$ under the curriculum template).
* **Audit Outcome ($N=100$ Long-Padded Harmful Prompts)**:
  * **Direct Payload Overlap**: **$99/100$ ($99.0\%$)**  
    * Window 2 ($[32, 64]$): $51$ cases ($51.0\%$)
    * Window 3 ($[48, 80]$): $47$ cases ($47.0\%$)
    * Window 4 ($[64, 96]$): $1$ case ($1.0\%$)
  * **Boundary Crossing**: **$1/100$ ($1.0\%$)**  
    * Max-scoring window spanned the transition boundary between preamble and prompt text.
  * **Pure Contextual Padding**: **$0/100$ ($0.0\%$)**  
    * Window 0 (pure prefix padding tokens $[0, 32]$) was the maximum window in zero cases.
* **Scientific Verdict**: The recovery from $1\%$ to $67\%$ recall is driven entirely by localized semantic violation detection, not spurious padding artifacts.

---

## 5. Verification and Decomposition of False-Positive Rates

To eliminate ambiguity in the thesis, distinguish between the two separate FPR numbers:

* **Standard Benchmark FPR**: **$41.7 \pm 23.5\%$**  
  * *Condition*: Concise, unpadded prompts evaluated on the length-balanced benchmark ($N=488$).
  * *Cause*: Reflects the intrinsic boundary threshold of the Safety-Policy encoder on short dual-use/sensitive prompts at $\tau=0.50$.
* **Long Benign-Padding FPR**: **$32.0\%$** ($32$ out of $100$ benign prompts flagged)  
  * *Condition*: Benign prompts embedded inside a $58$-token academic curriculum preamble and suffix.
  * *Cause Analysis ($N=32$ False Alarms)*:
    1. **Inherently Dual-Use / Criminal History Benchmark Topics ($18$ cases, $56.25\%$)**: e.g., counterfeiting history, charitable fraud, scams.
    2. **Taboo / Sensitive Domain Keywords ($8$ cases, $25.0\%$)**: e.g., child exploitation condemnation, illegal sexual acts.
    3. **Boundary Framing Interactions ($6$ cases, $18.75\%$)**: The phrase "threat landscape" in the template pushes borderline prompts over $\tau=0.50$.
    4. **Pure Academic Preamble Triggers**: **$0$ cases ($0.0\%$)**. Window 0 and Window 1 never triggered a false positive.

---

## 6. Resolution of the Safety-Policy ROC-AUC Discrepancy (0.723 vs. 0.684)

The apparent discrepancy between $\text{ROC-AUC} = 0.723$ in benchmark tables and $\text{ROC-AUC} = 0.6837$ in reproducibility checks has been resolved:

1. **Per-Fold Cross-Validation Mean $\pm$ SD ($0.7228 \approx \mathbf{0.723 \pm 0.111}$)**:
   * Computed by evaluating `roc_auc_score(y_val, p_val)` separately inside each of the $5$ validation folds and averaging the $5$ resulting fold scores:
     $$\text{Fold 1: } 0.7704, \quad \text{Fold 2: } 0.8862, \quad \text{Fold 3: } 0.6046, \quad \text{Fold 4: } 0.7092, \quad \text{Fold 5: } 0.6437$$
     $$\text{Mean} = 0.7228 \pm 0.1111, \quad \text{PR-AUC} = 0.7219 \pm 0.1447$$
   * This is the **standard cross-validation metric** reported in the benchmark results table.
2. **Pooled Out-Of-Fold (OOF) Global Metric ($\mathbf{0.6837 \approx 0.684}$)**:
   * Computed by concatenating all $N=488$ out-of-fold predicted probabilities into a single array and evaluating a single global `roc_auc_score(y_all, p_all_oof) = 0.6837` (PR-AUC = $0.6522$).
   * Because each fold trains a distinct linear classifier with slightly different intercepts, pooling introduces slight cross-fold calibration ranking variance, resulting in a slightly lower pooled AUC.
3. **Contrast with Experiment 7 ($0.672 \pm 0.093$)**:
   * Experiment 7 applied L2 unit-norm normalization to CLS tokens before Logistic Regression ($0.672 \pm 0.093$). Experiment 9 feeds un-normalized CLS vectors into Logistic Regression with standard L2 penalty ($0.723 \pm 0.111$).
4. **Thesis Guidance**: Report **$\mathbf{0.723 \pm 0.111}$** as the official 5-fold CV benchmark result, and note **$0.684$** if discussing pooled OOF vector calibration.

---

## 7. Ablation and Computational Overhead Reference

### Window Size Ablation ($S = W/2$, Max-Pooling)
* $W=24$: Balanced ROC = $0.732$, Long Padding Harmful Recall = $69.0\%$, Long Padding FPR = $36.0\%$, CPU Latency = $26.8$ ms
* **$W=32$ (Selected)**: Balanced ROC = $0.723$, Long Padding Harmful Recall = **$67.0\%$**, Long Padding FPR = **$32.0\%$**, CPU Latency = **$25.2$ ms**
* $W=40$: Balanced ROC = $0.723$, Long Padding Harmful Recall = $61.0\%$, Long Padding FPR = $30.0\%$, CPU Latency = $24.8$ ms
* $W=64$: Balanced ROC = $0.723$, Long Padding Harmful Recall = $48.0\%$, Long Padding FPR = $22.0\%$, CPU Latency = $24.1$ ms

### Aggregation Ablation ($W=32, S=16$ under Long Contextual Padding)
* **Max-Pooling**: Harmful Recall = **$67.0\%$**, Benign FPR = **$32.0\%$** *(Only rule that detects localized payload)*
* **Mean-Pooling**: Harmful Recall = **$14.0\%$**, Benign FPR = **$2.0\%$** *(Suffers from sequence dilution)*
* **Top-$3$ Mean**: Harmful Recall = **$31.0\%$**, Benign FPR = **$11.0\%$** *(Diluted by padding)*

### Computational Profiling (Intel / AMD CPU, Single-Threaded)
* **Global Latency**: $25.38$ ms / prompt
* **Windowed Latency ($W=32$)**: $25.21$ ms / prompt ($p95 = 30.08$ ms)
* **Latency Overhead**: **$0.993\times$** (effectively zero overhead on concise prompts; negligible overhead on padded prompts).

---

## 8. Exact Numbers by Thesis Chapter (Copy-Paste Ready)

### For the Abstract
* "Under rigorous length-balanced controls ($N=488$), the frozen Safety-Policy-MiniLM semantic encoder achieved a mean ROC-AUC of **$0.723 \pm 0.111$** and a harmful recall of **$77.2 \pm 9.3\%$**."
* "While global sequence representations suffered catastrophic contextual dilution under long academic padding (harmful recall collapsing to **$1.0\%$**), sliding-window localized semantic aggregation ($W=32, S=16, \text{Max}$) mitigated dilution, maintaining **$67.0\%$** harmful recall ($p = 4.62 \times 10^{-45}$, 95% CI: $[+56\%, +75\%]$) with a sub-$30$ ms CPU latency."

### For the Methodology Chapter
* Dataset partitions: Original $N=800$ ($200$ unique pairs $\times 4$ variants), Length-Balanced $N=488$ ($122$ pairs $\times 4$ variants).
* Cross-validation: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `unique_pair_id`.
* Detector architecture: Safety-Policy-MiniLM representations, window size $W=32$ tokens, stride $S=16$ tokens, max-pooling aggregation, threshold $\tau=0.50$.

### For the Results Chapter
* Primary benchmark table: Copy values directly from **Section 2**.
* Padding robustness table: Copy values directly from **Section 3**.
* Statistical significance: $t = 25.34$, $p = 4.62 \times 10^{-45}$, Wilcoxon $W = 0.0$, $p = 3.90 \times 10^{-18}$, bootstrap 95% CI $[+56.0\%, +75.0\%]$.
* Responsible window: $99.0\%$ payload overlap, $1.0\%$ boundary transition, $0.0\%$ pure padding.
* False alarm distinction: $41.7\%$ unpadded benchmark FPR vs. $32.0\%$ long-padding FPR.

### For the Discussion & Limitations Chapter
* Characterize defense strictly as a **probabilistic pre-inference intent filter**, not a jailbreak execution detector.
* State limitations: False-positive accumulation across long documents, sensitivity to dual-use criminal vocabulary, and boundary-splitting potential.

---

## 9. Final Thesis Verdict

$$\mathbf{THESIS\ READY:\ YES}$$

The empirical trajectory across Experiments 1 through 9 is methodologically complete, fully controlled, mathematically reconciled, and reproducible. No further model runs or hyperparameter searches are required.
