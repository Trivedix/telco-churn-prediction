"""
train.py
--------
Trains multiple ML models on the cleaned churn dataset, evaluates them,
picks the best performing model, and saves it to disk for the Flask app.

Models compared:
 - Logistic Regression
 - Random Forest
 - Gradient Boosting

Run this AFTER running clean.py (it depends on cleaned_churn.csv).
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from clean import CLEANED_FILE_PATH, TARGET_COL, MODELS_DIR

MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.pkl")

RANDOM_STATE = 42


def load_cleaned_data() -> pd.DataFrame:
    if not os.path.exists(CLEANED_FILE_PATH):
        raise FileNotFoundError(
            "❌ Cleaned dataset not found. Please run 'clean.py' first."
        )
    return pd.read_csv(CLEANED_FILE_PATH)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n----- {name} -----")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return metrics


def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_cleaned_data()
    X_train, X_test, y_train, y_test = split_data(df)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    best_model = None
    best_name = None
    best_score = -1
    all_metrics = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        all_metrics[name] = metrics

        # ROC-AUC is a solid single metric to rank churn models by
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model
            best_name = name

    print(f"\n🏆 Best model: {best_name} (ROC-AUC = {best_score:.4f})")

    # Save best model, feature column order, and metrics for the app/dashboard
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(list(X_train.columns), FEATURES_PATH)
    joblib.dump({"best_model": best_name, "all_metrics": all_metrics}, METRICS_PATH)

    print(f"✅ Best model saved to: {MODEL_PATH}")
    print(f"✅ Feature order saved to: {FEATURES_PATH}")

    return best_model, all_metrics


if __name__ == "__main__":
    train_models()
