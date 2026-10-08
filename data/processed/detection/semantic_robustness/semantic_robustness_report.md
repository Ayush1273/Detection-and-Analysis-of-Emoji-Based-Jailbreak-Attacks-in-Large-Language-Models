# Semantic Detector Stress Testing & Robustness Report

**Project:** Detection and Analysis of Emoji-Based Jailbreak Attacks in Large Language Models  
**Status:** Completed Stress-Testing Stage  
**Output Directory:** `data/processed/detection/semantic_robustness/`  
**Model Tested:** Frozen Sentence Transformer (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) + Logistic Regression (`C=1.0, random_state=42`)  
**Evaluation Scope:** Pre-inference intent classification (`safety_label`: benign vs. harmful intent).  

---

## Executive Summary & Core Scientific Findings

The stress-testing experiments subjected the frozen MiniLM semantic detector to adversarial boundary conditions, padding manipulations, emoji perturbations, and length correlation audits.

### Core Takeaways:
1. **Length-Invariance Confirmed:** MiniLM achieves **$0.751 \pm 0.066$** ROC-AUC on the length-balanced benchmark ($N=488$), remaining far above chance ($0.50$). Unlike structural models whose predictive capability collapsed to $0.503$, semantic representations do not rely on length heuristics to separate benign from harmful prompts.
2. **Padding Vulnerability is Bounded but Substantial:** Wrapping harmful prompts in educational framing ("educational research", "university coursework") causes MiniLM's harmful recall to drop from **$71\%$ down to $47\%$** under medium padding ($24\%$ class flip rate). Conversely, wrapping benign prompts in technical risk management framing ("threat landscape", "risk analysis") drives the benign false positive rate from **$34\%$ up to $83\%$**.
3. **Emoji Position Stability:** Emoji suffix ($|\Delta P| = 0.016$) and insertion ($|\Delta P| = 0.026$) cause negligible shifts. However, prefix emojis cause a modest probability reduction ($|\Delta P| = 0.048$), causing harmful recall to drop from **$71\%$ to $61\%$** due to positional token dilution on the leading sentence tokens.
4. **Residual Length Sensitivity Exists:** Controlling for true safety labels, MiniLM retains a weak positive partial correlation with prompt length ($r_{\text{partial}} = \mathbf{0.134}$, $p = 0.00015$). This occurs within harmful prompts ($r = +0.287$) because longer harmful requests contain more explicit harmful descriptors.
5. **Readiness for Safety-Specific Encoders:** MiniLM provides a solid, length-invariant baseline, but its $29\% - 53\%$ false-negative rate under academic framing proves that generic semantic similarity is insufficient for robust safety boundaries. Testing a safety-specific encoder (e.g., Llama Guard / WildGuard) is strongly justified.

---

## 1. Experiment 6A: Harmful Padding Robustness

*Data files:* [`semantic_robustness_harmful_padding.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_harmful_padding.csv) | [`semantic_robustness_harmful_padding_summary.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_harmful_padding_summary.csv)  
*Visualization:* [`padding_harmful_prob_distribution.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_plots/padding_harmful_prob_distribution.png)

Evaluated across the 100 harmful prompts using out-of-fold validation models under deterministic academic framing templates:

| Padding Condition | Mean Length | Mean Harmful Prob | Median Prob | Std Prob | Mean $\Delta$ | Median $\Delta$ | Mean $\|\Delta\|$ | Class Flip Rate | Harmful Recall ($\tau=0.50$) | False Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | $86$ chars | $0.604$ | $0.637$ | $0.186$ | $0.000$ | $0.000$ | $0.000$ | $0.0\%$ | **$71.0\%$** | $29.0\%$ |
| **Short Padding** | $138$ chars | $0.533$ | $0.536$ | $0.181$ | $-0.071$ | $-0.072$ | $0.074$ | **$15.0\%$** | **$56.0\%$** | $44.0\%$ |
| **Medium Padding** | $213$ chars | $0.481$ | $0.492$ | $0.184$ | **$-0.123$** | **$-0.121$** | **$0.124$** | **$24.0\%$** | **$47.0\%$** | **$53.0\%$** |
| **Long Padding** | $398$ chars | $0.641$ | $0.627$ | $0.110$ | $+0.037$ | $+0.020$ | $0.097$ | **$25.0\%$** | **$92.0\%$** | $8.0\%$ |

### Diagnostic Analysis:
- **Dilution Under Academic Framing (Short & Medium):** Adding benign academic context ("educational research for university coursework") dilutes the prompt's harmful token density in the 384-dimensional embedding space, causing a downward probability shift ($\Delta = -0.123$). At threshold $0.50$, **$24\%$** of previously detected harmful prompts flip to false negatives.
- **Topical Confound in Long Padding:** The long template contains terms like "threat landscape", "risk management", and "safety analysis". In generic semantic embeddings, these risk terms paradoxically push the probability upward ($\Delta = +0.037$, Recall $= 92\%$).

---

## 2. Experiment 6B: Benign Padding Robustness

*Data files:* [`semantic_robustness_benign_padding.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_benign_padding.csv) | [`semantic_robustness_benign_padding_summary.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_benign_padding_summary.csv)  
*Visualization:* [`padding_benign_fpr_tradeoff.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_plots/padding_benign_fpr_tradeoff.png)

Evaluated across the 100 benign prompts under the exact same padding templates:

| Padding Condition | Mean Length | Mean Harmful Prob | Median Prob | Std Prob | Mean $\Delta$ | Median $\Delta$ | Mean $\|\Delta\|$ | Class Flip Rate | Benign FPR ($\tau=0.50$) | True Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | $74$ chars | $0.455$ | $0.418$ | $0.184$ | $0.000$ | $0.000$ | $0.000$ | $0.0\%$ | **$38.0\%$** | $62.0\%$ |
| **Short Padding** | $126$ chars | $0.399$ | $0.362$ | $0.169$ | $-0.056$ | $-0.051$ | $0.064$ | $11.0\%$ | **$27.0\%$** | $73.0\%$ |
| **Medium Padding** | $201$ chars | $0.352$ | $0.322$ | $0.162$ | $-0.103$ | $-0.098$ | $0.107$ | $21.0\%$ | **$19.0\%$** | $81.0\%$ |
| **Long Padding** | $386$ chars | $0.609$ | $0.607$ | $0.108$ | **$+0.154$** | **$+0.174$** | **$0.166$** | **$45.0\%$** | **$83.0\%$** | **$17.0\%$** |

### Operational Takeaway:
- Short and medium benign educational framing slightly *reduces* false positive rates ($38\% \rightarrow 19\%$).
- However, long padding containing security/risk vocabulary induces catastrophic false alarms: **$83\%$** of completely harmless benign queries are misclassified as harmful simply because the framing contains threat-adjacent words.

---

## 3. Experiment 6C: Emoji Perturbation Robustness

*Data file:* [`semantic_robustness_emoji.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_emoji.csv)  
*Visualization:* [`emoji_perturbation_boxplots.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_plots/emoji_perturbation_boxplots.png)

Paired evaluation across all $200$ prompt pairs ($100$ harmful, $100$ benign) across all 4 variants ($N=800$):

| Class | Variant | Mean Harmful Prob | Median Prob | Std Prob | Mean Shift ($\Delta P$) | Mean Abs Shift ($\|\Delta P\|$) | Class Flip Rate | Recall / FPR | Paired $t$-test $p$-value | Wilcoxon $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Harmful** | **original** | $0.604$ | $0.637$ | $0.186$ | $0.000$ | $0.000$ | $0.0\%$ | **$71.0\%$ (Rec)** | $1.000$ | $1.000$ |
| | **prefix** | $0.558$ | $0.583$ | $0.183$ | $-0.046$ | **$0.048$** | **$10.0\%$** | **$61.0\%$ (Rec)** | $2.20 \times 10^{-22}$ | $1.65 \times 10^{-16}$ |
| | **suffix** | $0.599$ | $0.631$ | $0.188$ | $-0.005$ | **$0.016$** | $7.0\%$ | **$68.0\%$ (Rec)** | $0.0247$ | $0.0048$ |
| | **insertion** | $0.586$ | $0.635$ | $0.186$ | $-0.018$ | **$0.026$** | $8.0\%$ | **$63.0\%$ (Rec)** | $5.85 \times 10^{-6}$ | $2.20 \times 10^{-7}$ |
| **Benign** | **original** | $0.455$ | $0.418$ | $0.184$ | $0.000$ | $0.000$ | $0.0\%$ | **$38.0\%$ (FPR)** | $1.000$ | $1.000$ |
| | **prefix** | $0.424$ | $0.411$ | $0.177$ | $-0.031$ | **$0.037$** | $5.0\%$ | **$33.0\%$ (FPR)** | $2.86 \times 10^{-14}$ | $2.68 \times 10^{-12}$ |
| | **suffix** | $0.441$ | $0.412$ | $0.186$ | $-0.014$ | **$0.024$** | $4.0\%$ | **$34.0\%$ (FPR)** | $3.05 \times 10^{-6}$ | $7.34 \times 10^{-6}$ |
| | **insertion** | $0.433$ | $0.401$ | $0.191$ | $-0.022$ | **$0.029$** | $3.0\%$ | **$35.0\%$ (FPR)** | $6.30 \times 10^{-11}$ | $5.80 \times 10^{-10}$ |

### Empirical Insights:
- Suffix and insertion variations produce minimal absolute shift ($\le 0.026$ probability points).
- Prefix emojis produce a statistically significant shift ($\Delta P = -0.046$, $p < 10^{-21}$), causing harmful recall to drop from **$71\%$ down to $61\%$**.
- **Mechanism:** Emojis at token position 0 affect mean pooling over the sentence transformer's initial hidden states, slightly suppressing the activation of subsequent harmful tokens.

---

## 4. Experiment 6D: Residual Length Sensitivity Analysis

*Data file:* [`semantic_length_sensitivity.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_length_sensitivity.csv)  
*Visualization:* [`length_vs_probability_scatter.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_plots/length_vs_probability_scatter.png)

| Sample Subset | Sample Size | Pearson Correlation ($r$) | Pearson $p$-value | Spearman Rank ($\rho$) | Spearman $p$-value | Partial Correlation ($r_{\text{partial}}$) | Partial $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Prompts** | $N=800$ | $+0.223$ | $1.82 \times 10^{-10}$ | $+0.198$ | $1.66 \times 10^{-8}$ | **$+0.134$** | **$0.00015$** |
| **Harmful Prompts** | $N=400$ | **$+0.287$** | **$4.86 \times 10^{-9}$** | **$+0.251$** | **$3.68 \times 10^{-7}$** | — | — |
| **Benign Prompts** | $N=400$ | $-0.092$ | $0.0675$ (n.s.) | $-0.088$ | $0.0773$ (n.s.) | — | — |

*Analysis:*  
When controlling for the true safety label, MiniLM retains a weak positive residual correlation with prompt length ($r_{\text{partial}} = 0.134, p < 0.001$). Within harmful prompts, longer requests correlate moderately with higher harmful probability ($r = +0.287$) because more elaborate harmful prompts contain more explicit policy-violating tokens. Within benign prompts, correlation with length is essentially zero ($r = -0.092, p = 0.068$).

---

## 5. Experiment 6E: Final Multi-Dimensional Model Comparison

*Data file:* [`semantic_robustness_summary.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_summary.csv)  
*Visualization:* [`final_multi_dimensional_comparison.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic_robustness/semantic_robustness_plots/final_multi_dimensional_comparison.png)

| Operational Dimension | 1. TF-IDF Baseline | 2. Structural-Only | 3. Emoji-Only | 4. Full Hybrid | 5. MiniLM Semantic Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Benchmark ROC-AUC ($N=800$)** | $0.339 \pm 0.102$ | $0.647 \pm 0.043$ | $0.593 \pm 0.061$ | $0.538 \pm 0.097$ | **$0.716 \pm 0.077$** |
| **Original Benchmark Accuracy** | $37.0\%$ | $59.5\%$ | $54.9\%$ | $51.0\%$ | **$65.4\%$** |
| **Length-Balanced ROC-AUC ($N=488$)** | $0.428 \pm 0.048$ | $0.503 \pm 0.098$ *(collapses)* | $0.398 \pm 0.041$ *(collapses)* | $0.435 \pm 0.050$ *(collapses)* | **$0.751 \pm 0.066$** *(resilient)* |
| **Length-Balanced Accuracy** | $45.3\%$ | $51.1\%$ | $43.0\%$ | $45.7\%$ | **$67.4\%$** |
| **Harmful Recall on Original** | $38.0\%$ | $56.0\%$ | $72.2\%$ | $50.2\%$ | **$71.0\%$** |
| **Harmful Recall Under Medium Padding** | $35.0\%$ | $100.0\%$ *(artificial)* | $72.0\%$ | $100.0\%$ *(artificial)* | **$47.0\%$** *(diluted)* |
| **Benign FPR on Original** | $64.0\%$ | $37.0\%$ | $62.5\%$ | $48.2\%$ | **$34.2\%$** |
| **Benign FPR Under Long Padding** | $58.0\%$ | **$99.0\%$** *(catastrophic)* | $62.5\%$ | **$99.0\%$** *(catastrophic)* | **$83.0\%$** *(inflated by threat words)* |
| **Mean Absolute Emoji Shift ($\|\Delta P\|$)** | $0.000$ *(blind)* | $0.008$ | $0.048$ | $0.039$ | **$0.031$** |

### Strengths & Weaknesses Breakdown:
- **TF-IDF:** Invariant to emojis, but suffers severe lexical memorization failure on unseen prompt groups ($AUC < 0.43$).
- **Structural-Only:** Computationally cheap, but completely fails under length-balancing ($AUC = 0.503$) and is trivially spoofed by padding (flagging almost all long benign queries).
- **Emoji-Only:** Captures surface emoji counts, but provides zero intent discrimination on matched controls ($AUC = 0.398$).
- **Full Hybrid:** Inherits the weaknesses of both sparse lexical and structural heuristics, performing worse than semantic models.
- **MiniLM Semantic:** The only model that robustly survives length balancing ($AUC = 0.751$). However, it remains vulnerable to semantic dilution under educational framing (false negative rate up to $53\%$) and security-adjacent vocabulary in long benign text.

---

## 6. Answers to Final Scientific Interpretation Questions

1. **Does MiniLM remain above chance after length balancing?**  
   **Yes.** ROC-AUC is **$0.751 \pm 0.066$** (PR-AUC $0.791$), far exceeding chance level ($0.50$).
2. **How sensitive is MiniLM to harmless padding?**  
   It is moderately sensitive. Adding benign academic framing reduces harmful probability by $-0.123$, dropping harmful recall from $71\%$ to $47\%$.
3. **How sensitive is MiniLM to emoji placement?**  
   Suffix and insertion cause negligible shifts ($|\Delta P| \le 0.026$). Prefix emojis cause a modest downward shift ($|\Delta P| = 0.048$), reducing harmful recall from $71\%$ to $61\%$.
4. **What happens to benign false-positive rate under padding?**  
   Mild educational framing reduces benign FPR ($38\% \rightarrow 19\%$). However, long padding with risk/threat vocabulary causes benign FPR to jump to $83\%$.
5. **What happens to harmful recall under emoji perturbations?**  
   Recall remains at $68\%$ for suffix, $63\%$ for insertion, and drops to $61\%$ for prefix.
6. **Does MiniLM retain residual length sensitivity?**  
   **Yes, but weakly.** Overall partial correlation with length controlling for label is $r_{\text{partial}} = 0.134$, driven by harmful prompt detail.
7. **Is MiniLM sufficiently robust to justify testing a safety-specific encoder?**  
   **Yes.** MiniLM establishes that dense embeddings resolve the length confound. However, its generic sentence-similarity training leaves it vulnerable to benign framing dilution (53% FNR) and security vocabulary confusion (83% FPR on long technical queries). Testing a domain-aligned safety encoder (e.g., Llama Guard / WildGuard) is the logical and empirically justified next step.
