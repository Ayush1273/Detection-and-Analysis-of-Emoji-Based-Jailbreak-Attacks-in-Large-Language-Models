"""
Pre-Inference Jailbreak Detection Experiments
=============================================
Strictly pre-inference prompt-level adversarial defense filter.
Models:
  Model 1: Text Baseline (TF-IDF + Logistic Regression)
  Model 2: Emoji/Morphology Baseline (Features + Random Forest)
  Model 3: Proposed Hybrid (TF-IDF + Features + Logistic Regression)
Ablation Study:
  Config A: Text Only
  Config B: Emoji Mechanics Only
  Config C: Unicode Features Only
  Config D: Structural Features Only
  Config E: Non-Text Features (B + C + D)
  Config F: Full Hybrid (A + E)
Splitting:
  StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on unique_pair_id.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# Output directory
OUTPUT_DIR = Path("data/processed/detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = OUTPUT_DIR / "experiment.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("detector_experiment")


def verify_dataset(df: pd.DataFrame):
    """Verifies all Phase 1 dataset requirements programmatically."""
    logger.info("=== STEP 1: DATASET VERIFICATION ===")
    assert len(df) == 800, f"Expected 800 rows, got {len(df)}"
    logger.info(f"Row count verified: {len(df)}")
    
    assert "unique_pair_id" in df.columns, "unique_pair_id missing from dataset"
    n_groups = df["unique_pair_id"].nunique()
    assert n_groups == 200, f"Expected 200 unique_pair_id groups, got {n_groups}"
    logger.info(f"Group count verified: {n_groups} unique_pair_id groups")
    
    group_counts = df["unique_pair_id"].value_counts()
    assert (group_counts == 4).all(), "Every unique_pair_id must have exactly 4 variants"
    logger.info("Variant balance verified: Exactly 4 variants per unique_pair_id")
    
    assert "safety_label" in df.columns, "safety_label missing from dataset"
    label_counts = df["safety_label"].value_counts().to_dict()
    assert label_counts[0] == 400 and label_counts[1] == 400, f"Class imbalance detected: {label_counts}"
    logger.info("Class balance verified: 400 Benign (0), 400 Harmful (1)")
    
    # Verify absence of post-inference leakage
    prohibited_cols = [
        "response", "response_behavior", "generation_time_seconds",
        "preliminary_safety_outcome", "final_safety_outcome",
        "refusal_evidence", "harmful_content_evidence", "error"
    ]
    present_prohibited = [c for c in prohibited_cols if c in df.columns]
    assert len(present_prohibited) == 0, f"POST-INFERENCE LEAKAGE DETECTED! Prohibited columns present: {present_prohibited}"
    logger.info("Post-inference leakage check passed: 0 prohibited generation fields present")
    
    # Features required check
    required_features = [
        "prompt", "emoji_count", "emoji_density", "emoji_position",
        "detected_symbol_count", "unique_symbol_count", "unicode_categories",
        "text_length", "word_count", "space_count", "punctuation_count"
    ]
    nulls = df[required_features].isnull().sum()
    assert nulls.sum() == 0, f"Missing values in required features: {nulls[nulls > 0]}"
    logger.info("Feature completeness verified: 0 missing values in required feature set")


def verify_no_group_leakage(cv, X, y, groups):
    """Verifies that no unique_pair_id occurs in both train and validation folds."""
    logger.info("=== STEP 2: GROUP LEAKAGE VERIFICATION ===")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        leakage = train_groups.intersection(val_groups)
        assert len(leakage) == 0, f"CRITICAL: Fold {fold} has {len(leakage)} leaking groups!"
        logger.info(f"Fold {fold}: Train groups={len(train_groups)}, Val groups={len(val_groups)}, Leakage={len(leakage)} (PASSED)")


def evaluate_predictions(y_true, y_pred, y_prob):
    """Calculates all specified classification metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def run_experiments():
    # Load dataset
    df = pd.read_csv("data/processed/emoji_features.csv")
    verify_dataset(df)
    
    # Feature subsets
    TEXT_COL = "prompt"
    EMOJI_NUM = ["emoji_count", "emoji_density"]
    EMOJI_CAT = ["emoji_position"]
    UNICODE_NUM = ["detected_symbol_count", "unique_symbol_count"]
    UNICODE_CAT = ["unicode_categories"]
    STRUCT_NUM = ["text_length", "word_count", "space_count", "punctuation_count"]
    
    ALL_NUM = EMOJI_NUM + UNICODE_NUM + STRUCT_NUM
    ALL_CAT = EMOJI_CAT + UNICODE_CAT
    
    y = df["safety_label"]
    groups = df["unique_pair_id"]
    
    # Cross-validation
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    verify_no_group_leakage(cv, df, y, groups)
    
    # =========================================================================
    # MODEL DEFINITIONS
    # =========================================================================
    # Model 1: Text Baseline (TF-IDF + Logistic Regression)
    def build_model_1():
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words="english")),
            ("clf", LogisticRegression(C=1.0, solver="liblinear", random_state=42))
        ])
    
    # Model 2: Emoji/Morphology Baseline (Random Forest on Non-Text Features)
    def build_model_2():
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), ALL_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ALL_CAT)
        ])
        return Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=4, random_state=42))
        ])
    
    # Model 3: Proposed Hybrid (TF-IDF + Non-Text Features + Logistic Regression)
    def build_model_3():
        preprocessor = ColumnTransformer(transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words="english"), TEXT_COL),
            ("num", StandardScaler(), ALL_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ALL_CAT)
        ])
        return Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(C=1.0, solver="liblinear", random_state=42))
        ])
    
    # =========================================================================
    # ABLATION CONFIGURATIONS
    # =========================================================================
    def build_ablation_model(config_name):
        if config_name == "A_Text_Only":
            return build_model_1(), [TEXT_COL]
        elif config_name == "B_Emoji_Mechanics_Only":
            prep = ColumnTransformer([
                ("num", StandardScaler(), EMOJI_NUM),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), EMOJI_CAT)
            ])
            return Pipeline([("prep", prep), ("clf", LogisticRegression(C=1.0, solver="liblinear", random_state=42))]), EMOJI_NUM + EMOJI_CAT
        elif config_name == "C_Unicode_Only":
            prep = ColumnTransformer([
                ("num", StandardScaler(), UNICODE_NUM),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), UNICODE_CAT)
            ])
            return Pipeline([("prep", prep), ("clf", LogisticRegression(C=1.0, solver="liblinear", random_state=42))]), UNICODE_NUM + UNICODE_CAT
        elif config_name == "D_Structural_Only":
            prep = ColumnTransformer([
                ("num", StandardScaler(), STRUCT_NUM)
            ])
            return Pipeline([("prep", prep), ("clf", LogisticRegression(C=1.0, solver="liblinear", random_state=42))]), STRUCT_NUM
        elif config_name == "E_All_Non_Text":
            prep = ColumnTransformer([
                ("num", StandardScaler(), ALL_NUM),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ALL_CAT)
            ])
            return Pipeline([("prep", prep), ("clf", LogisticRegression(C=1.0, solver="liblinear", random_state=42))]), ALL_NUM + ALL_CAT
        elif config_name == "F_Full_Hybrid":
            return build_model_3(), [TEXT_COL] + ALL_NUM + ALL_CAT
        else:
            raise ValueError(f"Unknown config: {config_name}")

    # =========================================================================
    # RUN MAIN 3 MODELS
    # =========================================================================
    logger.info("=== STEP 3: RUNNING MAIN 3 MODELS (5-FOLD STRATIFIED GROUPED CV) ===")
    models = {
        "Model 1 (Text Baseline)": (build_model_1, [TEXT_COL]),
        "Model 2 (Emoji/Morphology Baseline)": (build_model_2, ALL_NUM + ALL_CAT),
        "Model 3 (Proposed Hybrid)": (build_model_3, [TEXT_COL] + ALL_NUM + ALL_CAT)
    }
    
    fold_records = []
    oof_predictions = {name: np.zeros(len(df)) for name in models}
    oof_probabilities = {name: np.zeros(len(df)) for name in models}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df, y, groups)):
        logger.info(f"--- FOLD {fold + 1} / 5 ---")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        for model_name, (builder, feat_cols) in models.items():
            model = builder()
            if len(feat_cols) == 1 and feat_cols[0] == TEXT_COL:
                X_train = train_df[TEXT_COL]
                X_val = val_df[TEXT_COL]
            else:
                X_train = train_df[feat_cols]
                X_val = val_df[feat_cols]
                
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            val_probs = model.predict_proba(X_val)[:, 1]
            
            oof_predictions[model_name][val_idx] = val_preds
            oof_probabilities[model_name][val_idx] = val_probs
            
            metrics = evaluate_predictions(y_val, val_preds, val_probs)
            record = {
                "fold": fold,
                "model": model_name,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                "cm_tn": metrics["confusion_matrix"][0][0],
                "cm_fp": metrics["confusion_matrix"][0][1],
                "cm_fn": metrics["confusion_matrix"][1][0],
                "cm_tp": metrics["confusion_matrix"][1][1]
            }
            fold_records.append(record)
            logger.info(f"[{model_name}] Fold {fold}: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(OUTPUT_DIR / "fold_level_results.csv", index=False)
    logger.info("Saved fold_level_results.csv")
    
    # Calculate fold-wise difference between Model 3 and Model 1 for each metric
    diff_records = []
    for f in range(5):
        m1_row = fold_df[(fold_df["fold"] == f) & (fold_df["model"] == "Model 1 (Text Baseline)")].iloc[0]
        m3_row = fold_df[(fold_df["fold"] == f) & (fold_df["model"] == "Model 3 (Proposed Hybrid)")].iloc[0]
        diff_record = {
            "fold": f,
            "roc_auc_diff_m3_minus_m1": m3_row["roc_auc"] - m1_row["roc_auc"],
            "pr_auc_diff_m3_minus_m1": m3_row["pr_auc"] - m1_row["pr_auc"],
            "f1_diff_m3_minus_m1": m3_row["f1"] - m1_row["f1"],
            "accuracy_diff_m3_minus_m1": m3_row["accuracy"] - m1_row["accuracy"],
            "balanced_accuracy_diff_m3_minus_m1": m3_row["balanced_accuracy"] - m1_row["balanced_accuracy"],
            "recall_diff_m3_minus_m1": m3_row["recall"] - m1_row["recall"],
            "precision_diff_m3_minus_m1": m3_row["precision"] - m1_row["precision"]
        }
        diff_records.append(diff_record)
    diff_df = pd.DataFrame(diff_records)
    diff_df.to_csv(OUTPUT_DIR / "fold_wise_differences_m3_vs_m1.csv", index=False)
    logger.info("Saved fold_wise_differences_m3_vs_m1.csv")

    # Save out-of-fold predictions
    oof_df = df[["variant_id", "unique_pair_id", "attack_type", "category", "safety_label"]].copy()
    for m in models:
        oof_df[f"{m}_pred"] = oof_predictions[m]
        oof_df[f"{m}_prob"] = oof_probabilities[m]
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    logger.info("Saved oof_predictions.csv")

    # Aggregate metrics across folds
    metric_cols = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    agg_summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
    agg_summary.to_csv(OUTPUT_DIR / "aggregate_results.csv")
    logger.info("Saved aggregate_results.csv")

    # =========================================================================
    # ABLATION EXPERIMENTS
    # =========================================================================
    logger.info("=== STEP 4: RUNNING ABLATION STUDY (CONFIGS A - F) ===")
    ablation_configs = [
        "A_Text_Only",
        "B_Emoji_Mechanics_Only",
        "C_Unicode_Only",
        "D_Structural_Only",
        "E_All_Non_Text",
        "F_Full_Hybrid"
    ]
    
    ablation_records = []
    for config_name in ablation_configs:
        logger.info(f"Running Ablation: {config_name}")
        for fold, (train_idx, val_idx) in enumerate(cv.split(df, y, groups)):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model, feat_cols = build_ablation_model(config_name)
            if len(feat_cols) == 1 and feat_cols[0] == TEXT_COL:
                X_train = train_df[TEXT_COL]
                X_val = val_df[TEXT_COL]
            else:
                X_train = train_df[feat_cols]
                X_val = val_df[feat_cols]
                
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            val_probs = model.predict_proba(X_val)[:, 1]
            
            metrics = evaluate_predictions(y_val, val_preds, val_probs)
            ablation_records.append({
                "config": config_name,
                "fold": fold,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"}
            })

    ablation_df = pd.DataFrame(ablation_records)
    ablation_df.to_csv(OUTPUT_DIR / "ablation_fold_level_results.csv", index=False)
    
    ablation_summary = ablation_df.groupby("config")[metric_cols].agg(["mean", "std"])
    ablation_summary.to_csv(OUTPUT_DIR / "ablation_summary_results.csv")
    logger.info("Saved ablation_summary_results.csv")

    # =========================================================================
    # GENERATE PUBLICATION-GRADE PLOTS
    # =========================================================================
    logger.info("=== STEP 5: GENERATING PLOTS ===")
    
    # 1. ROC Curves (Out-of-Fold)
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name in models:
        fpr, tpr, _ = roc_curve(y, oof_probabilities[model_name])
        auc_val = roc_auc_score(y, oof_probabilities[model_name])
        plt.plot(fpr, tpr, label=f"{model_name} (OOF AUC = {auc_val:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) - 5-Fold Stratified Grouped CV", fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves_models.png")
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name in models:
        prec, rec, _ = precision_recall_curve(y, oof_probabilities[model_name])
        ap_val = average_precision_score(y, oof_probabilities[model_name])
        plt.plot(rec, prec, label=f"{model_name} (PR-AUC = {ap_val:.3f})", lw=2)
    plt.axhline(0.5, color="k", linestyle="--", lw=1.5, label="Baseline (0.50)")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves - 5-Fold Stratified Grouped CV", fontsize=13)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curves_models.png")
    plt.close()
    
    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for idx, (model_name, ax) in enumerate(zip(models, axes)):
        cm = confusion_matrix(y, oof_predictions[model_name])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign (0)", "Harmful (1)"],
                    yticklabels=["Benign (0)", "Harmful (1)"])
        ax.set_title(f"{model_name}\nTotal OOF CM", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices_models.png")
    plt.close()

    # 4. Ablation Performance Comparison Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    abl_mean = ablation_df.groupby("config")["roc_auc"].mean().loc[ablation_configs]
    abl_std = ablation_df.groupby("config")["roc_auc"].std().loc[ablation_configs]
    x_labels = [c.replace("_", " ") for c in ablation_configs]
    plt.bar(x_labels, abl_mean, yerr=abl_std, capsize=5, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"], alpha=0.85)
    plt.ylabel("Mean ROC-AUC across 5 Folds", fontsize=11)
    plt.title("Feature Ablation Study: Predictive Capability of Feature Groups", fontsize=13)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.ylim(0.4, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_roc_auc_comparison.png")
    plt.close()
    
    logger.info("All plots generated and saved successfully to data/processed/detection/plots/")
    logger.info("=== EXPERIMENT RUN COMPLETE ===")

if __name__ == "__main__":
    run_experiments()
