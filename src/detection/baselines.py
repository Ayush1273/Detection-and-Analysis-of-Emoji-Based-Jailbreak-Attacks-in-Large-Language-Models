import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.utils.paths import EMOJI_FEATURES_CSV, RESULTS_DIR


def load_detection_data(csv_path: Path = EMOJI_FEATURES_CSV):
    """
    Loads dataset and formats binary classification target:
    emoji_attack_label:
      0 = original prompt (no emoji transformation)
      1 = emoji-transformed prompt (prefix, suffix, or insertion)
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["emoji_attack_label"] = (df["attack_type"] != "original").astype(int)
    return df


def run_baseline_experiments(
    df: pd.DataFrame = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> dict:
    """
    Reproduces and evaluates the baseline TF-IDF + Logistic Regression models:
      1. Default (unweighted) Logistic Regression
      2. Balanced Logistic Regression (class_weight='balanced')
    """
    if df is None:
        df = load_detection_data()

    X = df["prompt"]
    y = df["emoji_attack_label"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Train samples: {len(X_train_raw)} (Original: {(y_train == 0).sum()}, Attack: {(y_train == 1).sum()})")
    print(f"Test samples : {len(X_test_raw)} (Original: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()})")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"TF-IDF Train Matrix Shape: {X_train_tfidf.shape}")
    print(f"TF-IDF Test Matrix Shape : {X_test_tfidf.shape}")

    models = {
        "Logistic Regression (Default)": LogisticRegression(random_state=random_state),
        "Logistic Regression (Balanced)": LogisticRegression(class_weight="balanced", random_state=random_state)
    }

    results = {}

    for name, clf in models.items():
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": clf,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, zero_division=0)
        }

        print("\n" + "=" * 65)
        print(f"RESULTS: {name}")
        print("=" * 65)
        print(f"Accuracy         : {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Precision        : {prec:.4f}")
        print(f"Recall           : {rec:.4f}")
        print(f"F1 Score         : {f1:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN (Correct Original)   : {cm[0, 0]} | FP (Original as Attack): {cm[0, 1]}")
        print(f"  FN (Attack as Original) : {cm[1, 0]} | TP (Correct Attack)   : {cm[1, 1]}")

    return {
        "vectorizer": vectorizer,
        "splits": (X_train_raw, X_test_raw, y_train, y_test),
        "results": results
    }


if __name__ == "__main__":
    run_baseline_experiments()
