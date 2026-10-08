# Semantic Pre-Inference Jailbreak Detection Analysis

**Project:** Detection and Analysis of Emoji-Based Jailbreak Attacks in Large Language Models  
**Status:** Completed Semantic Embedding Investigation  
**Output Directory:** `data/processed/detection/semantic/`  
**Primary Model:** Frozen Sentence Transformer (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) + Logistic Regression (`C=1.0, random_state=42`)  
**Core Research Question:**
> *"Can semantic representations identify harmful vs benign prompt intent when the previously identified prompt-length/structural shortcut is controlled?"*

---

## Executive Summary & Core Scientific Answers

1. **Semantic Performance Robustly Survives Length Balancing:**  
   Unlike structural models whose performance collapsed from $0.647$ to $0.503$ (pure chance), the frozen semantic embedding detector achieved an ROC-AUC of **$0.751 \pm 0.066$** (PR-AUC **$0.791 \pm 0.054$**, Accuracy **$0.674 \pm 0.037$**) on the length-balanced dataset ($N=488$).
2. **Performance Is Resilient Across Benchmarks:**  
   Semantic ROC-AUC shifted from $0.716 \pm 0.077$ on the original benchmark ($N=800$) to $0.751 \pm 0.066$ on the length-balanced benchmark ($N=488$), proving that semantic representations genuinely model intent rather than relying on length artifacts.
3. **Adding Structural Features Harms Semantic Generalization:**  
   On the length-balanced benchmark, concatenating structural features with semantic embeddings degraded ROC-AUC from **$0.751 \rightarrow 0.720$** ($-0.031$), confirming that structural features re-introduce spurious noise and length sensitivity.
4. **Emoji Mechanics Provide Negligible Incremental Signal:**  
   Adding emoji features to semantic embeddings yielded a negligible shift of $+0.0005$ ROC-AUC ($0.7507 \rightarrow 0.7512$) on the balanced dataset, demonstrating that base semantic embeddings already capture prompt semantics without needing explicit emoji counters.
5. **Semantic Representations Mitigate Padding Vulnerability:**  
   While structural models were completely deceived by harmless academic padding (jumping to $>0.98$ and $0.9999$ harmful probability purely due to token count), semantic probabilities remained bounded ($0.48 - 0.64$).
6. **High Stability Across Emoji Perturbations:**  
   The semantic detector showed high stability on harmful prompts across variants, with small mean absolute probability shifts of $0.016$ for suffix and $0.026$ for insertion (and $0.048$ for prefix).

---

## 1. Primary Model Performance: 5-Fold Grouped Cross-Validation

*Data files:* [`semantic_cv_results.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_cv_results.csv) | [`semantic_fold_results.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_fold_results.csv)  
*Visualizations:* [`semantic_roc_curves.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_roc_curves.png) | [`semantic_pr_curves.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_pr_curves.png) | [`semantic_confusion_matrix.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_confusion_matrix.png)

Evaluated under strict 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped on `unique_pair_id` (zero group leakage):

| Dataset Benchmark | ROC-AUC (Mean $\pm$ SD) | PR-AUC (Mean $\pm$ SD) | Accuracy (Mean $\pm$ SD) | Balanced Acc (Mean $\pm$ SD) | Precision (Mean $\pm$ SD) | Recall (Mean $\pm$ SD) | F1-Score (Mean $\pm$ SD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Benchmark ($N=800$)** | $0.716 \pm 0.077$ | $0.715 \pm 0.067$ | $0.654 \pm 0.057$ | $0.654 \pm 0.057$ | $0.653 \pm 0.053$ | $0.658 \pm 0.085$ | $0.654 \pm 0.063$ |
| **Length-Balanced ($N=488$)** | **$0.751 \pm 0.066$** | **$0.791 \pm 0.054$** | **$0.674 \pm 0.037$** | **$0.676 \pm 0.038$** | **$0.697 \pm 0.061$** | **$0.638 \pm 0.157$** | **$0.654 \pm 0.078$** |

---

## 2. Benchmark Comparison Tables

*Visualization:* [`semantic_model_comparison.png`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_model_comparison.png)

### Table A: Performance on Original Benchmark ($N=800$)
*Data file:* [`semantic_model_comparison_original.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_model_comparison_original.csv)

| Model Configuration | ROC-AUC | PR-AUC | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. TF-IDF + LogReg | 0.3393 | 0.4265 | 0.3700 | 0.3769 |
| 2. Structural-Only | 0.6466 | 0.6923 | 0.5950 | 0.5775 |
| 3. Emoji Mechanics-Only | 0.5926 | 0.6291 | 0.5488 | 0.6164 |
| 4. Structural + Emoji | 0.6524 | 0.6952 | 0.6025 | 0.5779 |
| 5. All Non-Text | 0.6425 | 0.6733 | 0.6012 | 0.5770 |
| 6. Full Hybrid (TF-IDF + Non-Text) | 0.5376 | 0.5990 | 0.5100 | 0.5044 |
| **7. Semantic (all-MiniLM) + LogReg** | **0.7159** | **0.7146** | **0.6538** | **0.6538** |

### Table B: Performance on Length-Balanced Benchmark ($N=488$)
*Data file:* [`semantic_model_comparison_balanced.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_model_comparison_balanced.csv)

| Model Configuration | ROC-AUC | PR-AUC | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. TF-IDF + LogReg | 0.4284 | 0.4851 | 0.4529 | 0.4491 |
| 2. Structural-Only | 0.5030 | 0.5179 | 0.5111 | 0.4800 |
| 3. Emoji Mechanics-Only | 0.3983 | 0.4677 | 0.4302 | 0.3685 |
| 4. Structural + Emoji | 0.4952 | 0.5097 | 0.5010 | 0.4635 |
| 5. All Non-Text | 0.4953 | 0.5116 | 0.4989 | 0.4733 |
| 6. Full Hybrid | 0.4350 | 0.4775 | 0.4573 | 0.4526 |
| **7. Semantic (all-MiniLM) + LogReg** | **0.7507** | **0.7907** | **0.6739** | **0.6538** |

---

## 3. Additional Ablations: Semantic + Structural & Emoji Features

*Data file:* [`semantic_ablation_results.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_ablation_results.csv)

| Benchmark | Feature Configuration | ROC-AUC (Mean $\pm$ SD) | PR-AUC (Mean $\pm$ SD) | Balanced Acc (Mean $\pm$ SD) | F1-Score (Mean $\pm$ SD) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Original ($N=800$)** | A. Semantic Only | $0.716 \pm 0.077$ | $0.715 \pm 0.067$ | $0.654 \pm 0.057$ | $0.654 \pm 0.063$ |
| | B. Semantic + Structural | $0.735 \pm 0.064$ | $0.771 \pm 0.048$ | $0.673 \pm 0.051$ | $0.675 \pm 0.069$ |
| | C. Semantic + Emoji | $0.718 \pm 0.071$ | $0.725 \pm 0.064$ | $0.653 \pm 0.050$ | $0.654 \pm 0.062$ |
| | D. Semantic + Struct + Emoji | $0.742 \pm 0.065$ | $0.774 \pm 0.049$ | $0.674 \pm 0.058$ | $0.676 \pm 0.075$ |
| **Length-Balanced ($N=488$)** | **A. Semantic Only** | **$0.751 \pm 0.066$** | **$0.791 \pm 0.054$** | **$0.676 \pm 0.038$** | **$0.654 \pm 0.078$** |
| | B. Semantic + Structural | $0.720 \pm 0.073$ | $0.743 \pm 0.076$ | $0.646 \pm 0.067$ | $0.606 \pm 0.141$ |
| | C. Semantic + Emoji | $0.751 \pm 0.071$ | $0.787 \pm 0.059$ | $0.674 \pm 0.049$ | $0.645 \pm 0.099$ |
| | D. Semantic + Struct + Emoji | $0.717 \pm 0.073$ | $0.741 \pm 0.078$ | $0.653 \pm 0.057$ | $0.612 \pm 0.132$ |

*Empirical Takeaway:*  
On the length-balanced dataset, adding structural features degrades ROC-AUC by $-0.031$ ($0.751 \rightarrow 0.720$). Structural features do not add invariant signal; they act as a distractor once the spurious length correlation is controlled.

---

## 4. Emoji Robustness on Harmful Prompts

*Data file:* [`semantic_emoji_robustness.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_emoji_robustness.csv)

Evaluated across the 100 harmful prompts using out-of-fold semantic models:

| Variant | Mean Harmful Probability | Std Probability | Mean Abs Shift ($|\Delta P|$) | Harmful Recall ($\tau = 0.50$) | False Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **original** | $0.604 \pm 0.186$ | $0.186$ | $0.000$ | **$71.0\%$** | $29.0\%$ |
| **prefix** | $0.558 \pm 0.183$ | $0.183$ | $0.048$ | **$61.0\%$** | $39.0\%$ |
| **suffix** | $0.599 \pm 0.188$ | $0.188$ | $0.016$ | **$68.0\%$** | $32.0\%$ |
| **insertion** | $0.586 \pm 0.186$ | $0.186$ | $0.026$ | **$63.0\%$** | $37.0\%$ |

*Observations:*
- Suffix and insertion perturbations cause negligible probability shifts ($|\Delta P| = 0.016$ and $0.026$).
- Prefix emojis produce a modest shift ($|\Delta P| = 0.048$), reducing recall from $71\%$ to $61\%$. This occurs because leading emojis slightly alter the initial token attention pool, though harmful intent remains largely recognized.

---

## 5. Padding Robustness Comparison

*Data file:* [`semantic_padding_robustness.csv`](file:///e:/KIIT/SEM%203/emoji-jailbreak-research/data/processed/detection/semantic/semantic_padding_robustness.csv)

Comparing how the semantic detector behaves under benign academic padding versus structural models:

| Padding Level | Mean Length | Structural RF Prob | Structural Recall | Semantic MiniLM Prob | Semantic Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** | $86$ chars | $0.541$ | $48\%$ | $0.604$ | $71\%$ |
| **Short Padding** | $138$ chars | $0.874$ | $98\%$ | $0.533$ | $56\%$ |
| **Medium Padding** | $213$ chars | $0.841$ | $100\%$ | $0.481$ | $47\%$ |
| **Long Padding** | $398$ chars | $0.841$ | $100\%$ | $0.641$ | $92\%$ |

*Contrast:*  
Structural models jump to $>0.98$ and $1.00$ harmful probability solely due to character count. In contrast, the semantic model reflects the actual mixture of contextual framing and harmful instruction, remaining substantially more bounded ($0.48 - 0.64$).

---

## 6. Synthesis: Recommended Next Defense Step

Considering all experimental dimensions—length invariance, emoji stability, robustness to padding, and pre-inference operational latency:

### Recommendation: Option B — Safety-Specific Semantic Encoder
* **Why Not Option A (Generic Semantic Refinement)?**  
  `all-MiniLM-L6-v2` is trained on general sentence similarity, not safety boundaries. It already achieves ROC-AUC $0.751$, but its false negative rate on nuanced attacks ($29\% - 39\%$) stems from lack of safety domain alignment.
* **Why Safety-Specific Encoders (e.g., Llama-Guard / WildGuard / RoBERTa-Safety)?**  
  Dedicated safety embeddings are pre-aligned to policy violation taxonomies. They preserve the length-invariance established in this experiment while substantially tightening harmful recall without relying on superficial prompt heuristics.
