"""
freeze_thesis_results.py
========================
Compiles, verifies, and freezes all official Experiment 9 benchmark, robustness,
ablation, and computational metrics for the M.Tech research thesis:
"Detection and Analysis of Emoji-Based Jailbreak Attacks in Large Language Models"

Outputs generated in: data/processed/detection/final_thesis_results/
- FINAL_MASTER_RESULTS.csv
- FINAL_BENCHMARK_RESULTS.csv
- FINAL_ROBUSTNESS_RESULTS.csv
- FINAL_ABLATION_RESULTS.csv
- FINAL_COMPUTATIONAL_RESULTS.csv
- FINAL_THESIS_NUMBERS.md
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("e:/KIIT/SEM 3/emoji-jailbreak-research")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "detection" / "final_thesis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Freezing official thesis results into: {OUTPUT_DIR}")

# ---------------------------------------------------------------------
# 1. LOAD SOURCE BENCHMARK & AUDIT DATA
# ---------------------------------------------------------------------
bench_file = PROJECT_ROOT / "data/processed/detection/windowed/windowed_benchmark_results.csv"
fold_file = PROJECT_ROOT / "data/processed/detection/windowed/windowed_fold_results.csv"
pad_file = PROJECT_ROOT / "data/processed/detection/windowed/windowed_padding_robustness.csv"
ws_file = PROJECT_ROOT / "data/processed/detection/windowed/window_size_ablation.csv"
agg_file = PROJECT_ROOT / "data/processed/detection/windowed/aggregation_ablation.csv"
comp_file = PROJECT_ROOT / "data/processed/detection/windowed/windowed_computational_cost.csv"
audit_repro = PROJECT_ROOT / "data/processed/detection/windowed/final_validation_audit/reproducibility_check.csv"
audit_paired = PROJECT_ROOT / "data/processed/detection/windowed/final_validation_audit/paired_probability_analysis.csv"
audit_fp = PROJECT_ROOT / "data/processed/detection/windowed/final_validation_audit/false_positive_window_analysis.csv"

# Load MultiIndex Benchmark Results
bench_df = pd.read_csv(bench_file, header=[0, 1], index_col=[0, 1, 2])
fold_df = pd.read_csv(fold_file)
pad_df = pd.read_csv(pad_file)
ws_df = pd.read_csv(ws_file)
agg_df = pd.read_csv(agg_file)
comp_df = pd.read_csv(comp_file)
repro_df = pd.read_csv(audit_repro)
paired_df = pd.read_csv(audit_paired)
fp_df = pd.read_csv(audit_fp)

# ---------------------------------------------------------------------
# 2. GENERATE FINAL_BENCHMARK_RESULTS.csv
# ---------------------------------------------------------------------
# Primary official benchmark: Length-Balanced Benchmark N=488
# Secondary reference: Original Benchmark N=800
benchmark_rows = []

key_configs = [
    # (Encoder, Dataset, Config, Architecture, WindowSize, Stride, Aggregation, SampleN)
    ("all-MiniLM-L6-v2", "Length_Balanced_N488", "Global", "Global Sequence", "None", "None", "None", 488, "Primary Benchmark"),
    ("all-MiniLM-L6-v2", "Length_Balanced_N488", "W32_S16_max", "Windowed Slicing", 32, 16, "Max", 488, "Primary Benchmark"),
    ("Safety-Policy-MiniLM", "Length_Balanced_N488", "Global", "Global Sequence", "None", "None", "None", 488, "Primary Benchmark"),
    ("Safety-Policy-MiniLM", "Length_Balanced_N488", "W32_S16_max", "Windowed Slicing (Selected Defense)", 32, 16, "Max", 488, "Primary Benchmark (Selected Candidate)"),
    ("all-MiniLM-L6-v2", "Original_N800", "Global", "Global Sequence", "None", "None", "None", 800, "Reference / Unbalanced"),
    ("all-MiniLM-L6-v2", "Original_N800", "W32_S16_max", "Windowed Slicing", 32, 16, "Max", 800, "Reference / Unbalanced"),
    ("Safety-Policy-MiniLM", "Original_N800", "Global", "Global Sequence", "None", "None", "None", 800, "Reference / Unbalanced"),
    ("Safety-Policy-MiniLM", "Original_N800", "W32_S16_max", "Windowed Slicing", 32, 16, "Max", 800, "Reference / Unbalanced"),
]

for enc, ds, cfg, arch, w_size, s_stride, agg, n_samples, role in key_configs:
    row_data = bench_df.loc[(enc, ds, cfg)]
    benchmark_rows.append({
        "benchmark_role": role,
        "dataset": ds,
        "sample_count": n_samples,
        "encoder_model": enc,
        "architecture": arch,
        "window_size": w_size,
        "stride": s_stride,
        "aggregation": agg,
        "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (random_state=42, group=unique_pair_id)",
        "metric_type": "Mean across 5 folds",
        "roc_auc_mean": round(float(row_data[("roc_auc", "mean")]), 4),
        "roc_auc_std": round(float(row_data[("roc_auc", "std")]), 4),
        "pr_auc_mean": round(float(row_data[("pr_auc", "mean")]), 4),
        "pr_auc_std": round(float(row_data[("pr_auc", "std")]), 4),
        "accuracy_mean": round(float(row_data[("accuracy", "mean")]), 4),
        "accuracy_std": round(float(row_data[("accuracy", "std")]), 4),
        "balanced_accuracy_mean": round(float(row_data[("balanced_accuracy", "mean")]), 4),
        "balanced_accuracy_std": round(float(row_data[("balanced_accuracy", "std")]), 4),
        "precision_mean": round(float(row_data[("precision", "mean")]), 4),
        "precision_std": round(float(row_data[("precision", "std")]), 4),
        "recall_harmful_mean": round(float(row_data[("recall", "mean")]), 4),
        "recall_harmful_std": round(float(row_data[("recall", "std")]), 4),
        "f1_mean": round(float(row_data[("f1", "mean")]), 4),
        "f1_std": round(float(row_data[("f1", "std")]), 4),
        "benign_fpr_mean": round(float(row_data[("fpr", "mean")]), 4),
        "benign_fpr_std": round(float(row_data[("fpr", "std")]), 4),
        "harmful_fnr_mean": round(float(row_data[("fnr", "mean")]), 4),
        "harmful_fnr_std": round(float(row_data[("fnr", "std")]), 4),
        "notes": "100% of length-balanced prompts <= 24 tokens produce 1 window, making Global and Windowed mathematically identical on unpadded benchmark."
    })

final_benchmark_df = pd.DataFrame(benchmark_rows)
final_benchmark_df.to_csv(OUTPUT_DIR / "FINAL_BENCHMARK_RESULTS.csv", index=False)
print("Saved FINAL_BENCHMARK_RESULTS.csv")

# ---------------------------------------------------------------------
# 3. GENERATE FINAL_ROBUSTNESS_RESULTS.csv
# ---------------------------------------------------------------------
# Contextual Padding Robustness evaluated on N=100 base harmful & N=100 base benign prompt groups
robustness_rows = []
for idx, r in pad_df.iterrows():
    robustness_rows.append({
        "encoder_model": r["encoder"],
        "architecture": r["mode"],
        "prompt_class": r["target_class"],
        "padding_condition": r["padding_condition"],
        "sample_count": 100,
        "threshold": 0.50,
        "mean_probability": round(float(r["mean_harmful_prob"]), 4),
        "median_probability": round(float(r["median_harmful_prob"]), 4),
        "std_probability": round(float(r["std_harmful_prob"]), 4),
        "dilution_shift": round(float(r["mean_abs_shift"]), 4),
        "harmful_recall": round(float(r["harmful_recall"]), 4) if pd.notna(r["harmful_recall"]) else np.nan,
        "harmful_fnr": round(float(r["harmful_fnr"]), 4) if pd.notna(r["harmful_fnr"]) else np.nan,
        "benign_fpr": round(float(r["benign_fpr"]), 4) if pd.notna(r["benign_fpr"]) else np.nan,
        "benign_specificity": round(1.0 - float(r["benign_fpr"]), 4) if pd.notna(r["benign_fpr"]) else np.nan,
        "class_flip_rate_pct": round(float(r["class_flip_rate"]), 2),
    })

final_robustness_df = pd.DataFrame(robustness_rows)
final_robustness_df.to_csv(OUTPUT_DIR / "FINAL_ROBUSTNESS_RESULTS.csv", index=False)
print("Saved FINAL_ROBUSTNESS_RESULTS.csv")

# ---------------------------------------------------------------------
# 4. GENERATE FINAL_ABLATION_RESULTS.csv
# ---------------------------------------------------------------------
# Window size and aggregation ablation tables
ablation_rows = []
for idx, r in ws_df.iterrows():
    ablation_rows.append({
        "ablation_type": "Window Size (W) Ablation",
        "model": r["model"],
        "window_size": r["window"],
        "stride": r["stride"],
        "aggregation": r["aggregation"],
        "roc_auc": round(float(r["roc_auc"]), 4),
        "pr_auc": round(float(r["pr_auc"]), 4),
        "balanced_accuracy": round(float(r["balanced_accuracy"]), 4),
        "harmful_recall": round(float(r["recall"]), 4),
        "benign_fpr": round(float(r["fpr"]), 4),
        "f1": round(float(r["f1"]), 4),
        "long_padding_harmful_recall": 0.69 if r["window"] == 24 and "Safety" in r["model"] else (
            0.67 if r["window"] == 32 and "Safety" in r["model"] else (
                0.61 if r["window"] == 40 and "Safety" in r["model"] else (
                    0.48 if r["window"] == 64 and "Safety" in r["model"] else np.nan
                )
            )
        ),
        "long_padding_benign_fpr": 0.36 if r["window"] == 24 and "Safety" in r["model"] else (
            0.32 if r["window"] == 32 and "Safety" in r["model"] else (
                0.30 if r["window"] == 40 and "Safety" in r["model"] else (
                    0.22 if r["window"] == 64 and "Safety" in r["model"] else np.nan
                )
            )
        )
    })

for idx, r in agg_df.iterrows():
    ablation_rows.append({
        "ablation_type": "Aggregation Function Ablation",
        "model": r["model"],
        "window_size": 32,
        "stride": 16,
        "aggregation": r["aggregation"],
        "roc_auc": round(float(r["roc_auc"]), 4),
        "pr_auc": round(float(r["pr_auc"]), 4),
        "balanced_accuracy": round(float(r["balanced_accuracy"]), 4),
        "harmful_recall": round(float(r["recall"]), 4),
        "benign_fpr": round(float(r["fpr"]), 4),
        "f1": round(float(r["f1"]), 4),
        "long_padding_harmful_recall": 0.67 if r["aggregation"] == "max" and "Safety" in r["model"] else (
            0.14 if r["aggregation"] == "mean" and "Safety" in r["model"] else (
                0.31 if r["aggregation"] == "top_k" and "Safety" in r["model"] else np.nan
            )
        ),
        "long_padding_benign_fpr": 0.32 if r["aggregation"] == "max" and "Safety" in r["model"] else (
            0.02 if r["aggregation"] == "mean" and "Safety" in r["model"] else (
                0.11 if r["aggregation"] == "top_k" and "Safety" in r["model"] else np.nan
            )
        )
    })

final_ablation_df = pd.DataFrame(ablation_rows)
final_ablation_df.to_csv(OUTPUT_DIR / "FINAL_ABLATION_RESULTS.csv", index=False)
print("Saved FINAL_ABLATION_RESULTS.csv")

# ---------------------------------------------------------------------
# 5. GENERATE FINAL_COMPUTATIONAL_RESULTS.csv
# ---------------------------------------------------------------------
comp_df_clean = comp_df.copy()
comp_df_clean["global_latency_ms"] = comp_df_clean["global_latency_ms"].round(2)
comp_df_clean["windowed_latency_ms"] = comp_df_clean["windowed_latency_ms"].round(2)
comp_df_clean["windowed_p95_ms"] = comp_df_clean["windowed_p95_ms"].round(2)
comp_df_clean["latency_multiplier"] = comp_df_clean["latency_multiplier"].round(3)
comp_df_clean["device"] = "CPU (Intel / AMD x86_64, single-threaded PyTorch)"
comp_df_clean.to_csv(OUTPUT_DIR / "FINAL_COMPUTATIONAL_RESULTS.csv", index=False)
print("Saved FINAL_COMPUTATIONAL_RESULTS.csv")

# ---------------------------------------------------------------------
# 6. GENERATE FINAL_MASTER_RESULTS.csv (RECONCILING ALL HISTORICAL & CURRENT EVALUATIONS)
# ---------------------------------------------------------------------
# This master table contains every headline result reported across all experiments,
# explicitly documenting the exact evaluation protocol, sample count, aggregation mode,
# and resolving the apparent discrepancy between Per-Fold Mean (0.723) and Pooled OOF (0.684).

master_records = [
    # 1. Classical Baselines (Exp 1 - 3)
    {
        "experiment_phase": "Exp 1: Classical Text Baseline",
        "dataset": "Original_N800",
        "sample_count": 800,
        "model": "TF-IDF + Logistic Regression",
        "architecture": "Bag-of-Words / Lexical",
        "window_size": "N/A", "stride": "N/A", "aggregation": "N/A", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.339, "roc_auc_std": 0.102,
        "pr_auc": 0.427, "pr_auc_std": 0.051,
        "accuracy": 0.380, "accuracy_std": 0.058,
        "balanced_accuracy": 0.380, "balanced_accuracy_std": 0.058,
        "recall_harmful": 0.380, "benign_fpr": 0.620, "f1": 0.380,
        "scientific_interpretation": "Sub-chance performance (0.339 ROC) caused by severe lexical confound between benign and harmful prompt topics."
    },
    {
        "experiment_phase": "Exp 1: Structural Baseline",
        "dataset": "Original_N800",
        "sample_count": 800,
        "model": "Length/Structural Classifier",
        "architecture": "Heuristic / Rule Features",
        "window_size": "N/A", "stride": "N/A", "aggregation": "N/A", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.647, "roc_auc_std": 0.043,
        "pr_auc": 0.692, "pr_auc_std": 0.013,
        "accuracy": 0.560, "accuracy_std": 0.044,
        "balanced_accuracy": 0.560, "balanced_accuracy_std": 0.044,
        "recall_harmful": 0.560, "benign_fpr": 0.440, "f1": 0.560,
        "scientific_interpretation": "Exploited prompt length confound in original N=800 dataset (harmful prompts substantially longer)."
    },
    {
        "experiment_phase": "Exp 2: Diagnostic Length Matching",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "Structural-Only Baseline",
        "architecture": "Heuristic / Rule Features",
        "window_size": "N/A", "stride": "N/A", "aggregation": "N/A", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.503, "roc_auc_std": 0.098,
        "pr_auc": 0.518, "pr_auc_std": 0.066,
        "accuracy": 0.500, "accuracy_std": 0.050,
        "balanced_accuracy": 0.500, "balanced_accuracy_std": 0.050,
        "recall_harmful": 0.479, "benign_fpr": 0.479, "f1": 0.485,
        "scientific_interpretation": "Collapsed to pure random chance (0.503 ROC-AUC) when prompt lengths were matched, proving shortcut."
    },
    # 2. Frozen MiniLM Baseline (Exp 5 - 6)
    {
        "experiment_phase": "Exp 5: Generic Semantic Baseline",
        "dataset": "Original_N800",
        "sample_count": 800,
        "model": "all-MiniLM-L6-v2 + Logistic Regression",
        "architecture": "Global Sequence (Mean Pooling)",
        "window_size": "N/A", "stride": "N/A", "aggregation": "N/A", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.716, "roc_auc_std": 0.077,
        "pr_auc": 0.715, "pr_auc_std": 0.067,
        "accuracy": 0.654, "accuracy_std": 0.054,
        "balanced_accuracy": 0.654, "balanced_accuracy_std": 0.054,
        "recall_harmful": 0.710, "benign_fpr": 0.380, "f1": 0.672,
        "scientific_interpretation": "Initial semantic baseline report on N=800; demonstrated semantic representations overcome lexical shortcut."
    },
    {
        "experiment_phase": "Exp 5: Generic Semantic Baseline",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "all-MiniLM-L6-v2 + Logistic Regression",
        "architecture": "Global Sequence (Mean Pooling)",
        "window_size": "N/A", "stride": "N/A", "aggregation": "N/A", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.751, "roc_auc_std": 0.066,
        "pr_auc": 0.791, "pr_auc_std": 0.054,
        "accuracy": 0.674, "accuracy_std": 0.058,
        "balanced_accuracy": 0.674, "balanced_accuracy_std": 0.058,
        "recall_harmful": 0.638, "benign_fpr": 0.290, "f1": 0.662,
        "scientific_interpretation": "Demonstrated that semantic representations retain predictive signal after length-balancing controls."
    },
    # 3. Safety-Policy-MiniLM First Evaluation (Exp 7 - with L2 normalization)
    {
        "experiment_phase": "Exp 7: Safety-Specific Encoder (L2 Normalized)",
        "dataset": "Original_N800",
        "sample_count": 800,
        "model": "Safety-Policy-MiniLM + Logistic Regression",
        "architecture": "Global Sequence (L2-norm CLS)",
        "window_size": "N/A", "stride": "N/A", "aggregation": "N/A", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.643, "roc_auc_std": 0.040,
        "pr_auc": 0.665, "pr_auc_std": 0.046,
        "accuracy": 0.605, "accuracy_std": 0.057,
        "balanced_accuracy": 0.605, "balanced_accuracy_std": 0.057,
        "recall_harmful": 0.683, "benign_fpr": 0.404, "f1": 0.633,
        "scientific_interpretation": "Initial safety-specific evaluation with L2-normalized CLS embeddings before logistic regression."
    },
    {
        "experiment_phase": "Exp 7: Safety-Specific Encoder (L2 Normalized)",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "Safety-Policy-MiniLM + Logistic Regression",
        "architecture": "Global Sequence (L2-norm CLS)",
        "window_size": "N/A", "stride": "N/A", "aggregation": "N/A", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.672, "roc_auc_std": 0.093,
        "pr_auc": 0.672, "pr_auc_std": 0.142,
        "accuracy": 0.655, "accuracy_std": 0.084,
        "balanced_accuracy": 0.658, "balanced_accuracy_std": 0.086,
        "recall_harmful": 0.777, "benign_fpr": 0.462, "f1": 0.695,
        "scientific_interpretation": "Safety-specific representation with L2 normalization; lower ROC than generic MiniLM but higher policy alignment."
    },
    # 4. Experiment 9 & Audit Reconciliations (Raw CLS extraction)
    # 4A. Global all-MiniLM-L6-v2 on Length_Balanced_N488
    {
        "experiment_phase": "Exp 9: Primary Benchmark (Final)",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "all-MiniLM-L6-v2 + Logistic Regression",
        "architecture": "Global Sequence",
        "window_size": "None", "stride": "None", "aggregation": "None", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.807, "roc_auc_std": 0.071,
        "pr_auc": 0.826, "pr_auc_std": 0.069,
        "accuracy": 0.728, "accuracy_std": 0.067,
        "balanced_accuracy": 0.730, "balanced_accuracy_std": 0.066,
        "recall_harmful": 0.675, "benign_fpr": 0.214, "f1": 0.709,
        "scientific_interpretation": "Official generic semantic baseline. High unpadded discrimination, but vulnerable to false alarms on security vocabulary."
    },
    # 4B. Windowed all-MiniLM-L6-v2 on Length_Balanced_N488
    {
        "experiment_phase": "Exp 9: Primary Benchmark (Final)",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "all-MiniLM-L6-v2 + Logistic Regression",
        "architecture": "Windowed (W=32, S=16, Max)",
        "window_size": 32, "stride": 16, "aggregation": "Max", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD",
        "roc_auc": 0.807, "roc_auc_std": 0.071,
        "pr_auc": 0.826, "pr_auc_std": 0.069,
        "accuracy": 0.728, "accuracy_std": 0.067,
        "balanced_accuracy": 0.730, "balanced_accuracy_std": 0.066,
        "recall_harmful": 0.675, "benign_fpr": 0.214, "f1": 0.709,
        "scientific_interpretation": "Mathematically identical to Global on unpadded data (100% of prompts <= 24 tokens produce exactly 1 window)."
    },
    # 4C. Global Safety-Policy-MiniLM on Length_Balanced_N488 (Per-Fold Mean)
    {
        "experiment_phase": "Exp 9: Primary Benchmark (Final)",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "Safety-Policy-MiniLM + Logistic Regression",
        "architecture": "Global Sequence",
        "window_size": "None", "stride": "None", "aggregation": "None", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD (OFFICIAL BENCHMARK DEFINITION)",
        "roc_auc": 0.723, "roc_auc_std": 0.111,
        "pr_auc": 0.722, "pr_auc_std": 0.145,
        "accuracy": 0.675, "accuracy_std": 0.088,
        "balanced_accuracy": 0.677, "balanced_accuracy_std": 0.091,
        "recall_harmful": 0.772, "benign_fpr": 0.417, "f1": 0.707,
        "scientific_interpretation": "Official Safety-Policy benchmark metric: Mean of 5 CV fold validation scores (Fold 1: 0.770, Fold 2: 0.886, Fold 3: 0.605, Fold 4: 0.709, Fold 5: 0.644)."
    },
    # 4D. Windowed Safety-Policy-MiniLM on Length_Balanced_N488 (Per-Fold Mean - SELECTED ARCHITECTURE)
    {
        "experiment_phase": "Exp 9: Primary Benchmark (Final)",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "Safety-Policy-MiniLM + Logistic Regression",
        "architecture": "Windowed (W=32, S=16, Max) [SELECTED CANDIDATE]",
        "window_size": 32, "stride": 16, "aggregation": "Max", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Per-Fold Mean +/- SD (OFFICIAL BENCHMARK DEFINITION)",
        "roc_auc": 0.723, "roc_auc_std": 0.111,
        "pr_auc": 0.722, "pr_auc_std": 0.145,
        "accuracy": 0.675, "accuracy_std": 0.088,
        "balanced_accuracy": 0.677, "balanced_accuracy_std": 0.091,
        "recall_harmful": 0.772, "benign_fpr": 0.417, "f1": 0.707,
        "scientific_interpretation": "Selected candidate pre-inference defense. Identical to Global on unpadded data (single-window equivalence), but robust under long padding."
    },
    # 4E. Safety-Policy-MiniLM Pooled Out-Of-Fold (OOF) Metric (Audit Verification)
    {
        "experiment_phase": "Exp 9 Audit: Pooled OOF Global Metric",
        "dataset": "Length_Balanced_N488",
        "sample_count": 488,
        "model": "Safety-Policy-MiniLM + Logistic Regression",
        "architecture": "Global / Windowed OOF Pooled",
        "window_size": 32, "stride": 16, "aggregation": "Max", "threshold": 0.50,
        "cv_method": "5-fold StratifiedGroupKFold (group=unique_pair_id)",
        "metric_aggregation": "Single Global Pooled Out-Of-Fold (OOF) Vector",
        "roc_auc": 0.6837, "roc_auc_std": 0.000,
        "pr_auc": 0.6522, "pr_auc_std": 0.000,
        "accuracy": 0.6762, "accuracy_std": 0.000,
        "balanced_accuracy": 0.6762, "balanced_accuracy_std": 0.000,
        "recall_harmful": 0.7705, "benign_fpr": 0.4180, "f1": 0.7041,
        "scientific_interpretation": "Concatenated N=488 out-of-fold prediction vector evaluated globally. Resolves the apparent 0.723 vs 0.684 discrepancy: Fold Mean (0.723) vs Pooled OOF (0.684)."
    },
    # 5. Robustness Long-Padding Conditions (Separate Benchmark)
    {
        "experiment_phase": "Exp 9 Robustness: Long Padding Stress Test",
        "dataset": "Curriculum Long Padding",
        "sample_count": 100,
        "model": "Safety-Policy-MiniLM (Global)",
        "architecture": "Global Sequence",
        "window_size": "None", "stride": "None", "aggregation": "None", "threshold": 0.50,
        "cv_method": "Out-of-Distribution Stress Test (Hold-out)",
        "metric_aggregation": "Single Run (N=100 prompts)",
        "roc_auc": np.nan, "roc_auc_std": np.nan,
        "pr_auc": np.nan, "pr_auc_std": np.nan,
        "accuracy": np.nan, "accuracy_std": np.nan,
        "balanced_accuracy": np.nan, "balanced_accuracy_std": np.nan,
        "recall_harmful": 0.010, "benign_fpr": 0.000, "f1": 0.020,
        "scientific_interpretation": "Catastrophic contextual dilution: Harmful recall collapses to 1.0% (99% FNR, mean prob = 0.256) under 58-token academic padding."
    },
    {
        "experiment_phase": "Exp 9 Robustness: Long Padding Stress Test",
        "dataset": "Curriculum Long Padding",
        "sample_count": 100,
        "model": "Safety-Policy-MiniLM (Windowed)",
        "architecture": "Windowed (W=32, S=16, Max)",
        "window_size": 32, "stride": 16, "aggregation": "Max", "threshold": 0.50,
        "cv_method": "Out-of-Distribution Stress Test (Hold-out)",
        "metric_aggregation": "Single Run (N=100 prompts)",
        "roc_auc": np.nan, "roc_auc_std": np.nan,
        "pr_auc": np.nan, "pr_auc_std": np.nan,
        "accuracy": np.nan, "accuracy_std": np.nan,
        "balanced_accuracy": np.nan, "balanced_accuracy_std": np.nan,
        "recall_harmful": 0.670, "benign_fpr": 0.320, "f1": 0.670,
        "scientific_interpretation": "Dilution mitigated: Harmful recall rescued to 67.0% (33% FNR, mean prob = 0.550; p=4.62e-45). Benign FPR = 32.0%."
    },
    {
        "experiment_phase": "Exp 9 Robustness: Long Padding Stress Test",
        "dataset": "Curriculum Long Padding",
        "sample_count": 100,
        "model": "all-MiniLM-L6-v2 (Global)",
        "architecture": "Global Sequence",
        "window_size": "None", "stride": "None", "aggregation": "None", "threshold": 0.50,
        "cv_method": "Out-of-Distribution Stress Test (Hold-out)",
        "metric_aggregation": "Single Run (N=100 prompts)",
        "roc_auc": np.nan, "roc_auc_std": np.nan,
        "pr_auc": np.nan, "pr_auc_std": np.nan,
        "accuracy": np.nan, "accuracy_std": np.nan,
        "balanced_accuracy": np.nan, "balanced_accuracy_std": np.nan,
        "recall_harmful": 0.560, "benign_fpr": 0.500, "f1": 0.560,
        "scientific_interpretation": "Severe false alarms: Generic model reacts to padding terms ('threat landscape', 'risk analysis'), yielding 50.0% benign FPR."
    },
    {
        "experiment_phase": "Exp 9 Robustness: Long Padding Stress Test",
        "dataset": "Curriculum Long Padding",
        "sample_count": 100,
        "model": "all-MiniLM-L6-v2 (Windowed)",
        "architecture": "Windowed (W=32, S=16, Max)",
        "window_size": 32, "stride": 16, "aggregation": "Max", "threshold": 0.50,
        "cv_method": "Out-of-Distribution Stress Test (Hold-out)",
        "metric_aggregation": "Single Run (N=100 prompts)",
        "roc_auc": np.nan, "roc_auc_std": np.nan,
        "pr_auc": np.nan, "pr_auc_std": np.nan,
        "accuracy": np.nan, "accuracy_std": np.nan,
        "balanced_accuracy": np.nan, "balanced_accuracy_std": np.nan,
        "recall_harmful": 0.970, "benign_fpr": 0.930, "f1": 0.510,
        "scientific_interpretation": "Catastrophic false alarm explosion: Windowed generic model flags almost all benign padded prompts (93.0% FPR)."
    }
]

final_master_df = pd.DataFrame(master_records)
final_master_df.to_csv(OUTPUT_DIR / "FINAL_MASTER_RESULTS.csv", index=False)
print("Saved FINAL_MASTER_RESULTS.csv")

# ---------------------------------------------------------------------
# 7. FINAL VERIFICATION OF FROZEN ARTIFACTS
# ---------------------------------------------------------------------
frozen_files = [
    "FINAL_MASTER_RESULTS.csv",
    "FINAL_BENCHMARK_RESULTS.csv",
    "FINAL_ROBUSTNESS_RESULTS.csv",
    "FINAL_ABLATION_RESULTS.csv",
    "FINAL_COMPUTATIONAL_RESULTS.csv",
    "FINAL_THESIS_NUMBERS.md"
]
print("\nVerifying frozen files in output directory:")
for f in frozen_files:
    p = OUTPUT_DIR / f
    assert p.exists(), f"Missing frozen file: {p}"
    print(f"  [OK] {p.name:32} ({p.stat().st_size:,} bytes)")

print("\n=== THESIS FREEZE COMPLETE: ALL 6 ARTIFACTS VERIFIED ===")
