"""
modelling.py
============
Baseline model training with MLflow experiment tracking.
Trains 3 models: LogisticRegression, RandomForest, GradientBoosting
and logs all artifacts to MLflow Tracking Server.

Usage:
    1. Start MLflow server:  mlflow server --host 127.0.0.1 --port 5000
    2. Run:                  python modelling.py
    3. View UI:              http://127.0.0.1:5000
"""

import os
import json
import warnings

import dagshub
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from wine_quality_preprocessing import load_data, preprocess

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
EXPERIMENT_NAME = "wine_quality_baseline"

dagshub.init(
    repo_owner="alsyamaulizar",
    repo_name="wine-quality-ml",
    mlflow=True
)


# ── Helper functions ──────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Bad", "Good"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    path = f"cm_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(importances)))
    ax.bar(range(len(importances)), importances[indices], color=colors)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices], rotation=45, ha="right", fontsize=9
    )
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13)
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = f"feature_importance_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return path


def plot_roc_curve(model, X_test, y_test, model_name: str) -> str:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2.5,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = f"roc_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return path


# ── Main training function ────────────────────────────────────
def train_and_log(
    model,
    model_name: str,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    feature_names: list,
) -> tuple:
    with mlflow.start_run(run_name=model_name) as run:
        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_val_pred  = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "val_accuracy" : accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
            "val_recall"   : recall_score(y_val, y_val_pred, zero_division=0),
            "val_f1"       : f1_score(y_val, y_val_pred, zero_division=0),
            "val_roc_auc"  : roc_auc_score(y_val, y_val_proba),
            "test_accuracy" : accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall"   : recall_score(y_test, y_test_pred, zero_division=0),
            "test_f1"       : f1_score(y_test, y_test_pred, zero_division=0),
            "test_roc_auc"  : roc_auc_score(y_test, y_test_proba),
        }

        # Log params & metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model_type": type(model).__name__, "dataset": "wine_quality"})

        # ── Artifacts ──────────────────────────────────────
        # 1. Confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_test_pred, model_name)
        mlflow.log_artifact(cm_path)

        # 2. Feature importance (tree models)
        fi_path = plot_feature_importance(model, feature_names, model_name)
        if fi_path:
            mlflow.log_artifact(fi_path)

        # 3. ROC Curve
        roc_path = plot_roc_curve(model, X_test, y_test, model_name)
        mlflow.log_artifact(roc_path)

        # 4. Classification report
        report = classification_report(y_test, y_test_pred, target_names=["Bad", "Good"])
        report_path = f"classification_report_{model_name}.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n\n{report}")
        mlflow.log_artifact(report_path)

        # 5. Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name,
            registered_model_name=f"WineQuality_{model_name}",
        )

        print(f"\n{'─'*55}")
        print(f"  Model       : {model_name}")
        print(f"  Val  F1     : {metrics['val_f1']:.4f}  | AUC: {metrics['val_roc_auc']:.4f}")
        print(f"  Test F1     : {metrics['test_f1']:.4f}  | AUC: {metrics['test_roc_auc']:.4f}")
        print(f"  Run ID      : {run.info.run_id}")

        return metrics["test_f1"], run.info.run_id


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("   Wine Quality — Baseline Model Training")
    print("=" * 55)

    print("Tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = preprocess(df)

    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    }

    results = {}

    for name, model in models.items():
        f1, run_id = train_and_log(
            model,
            name,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            features
        )

        results[name] = {
            "f1": float(f1),
            "run_id": str(run_id)
        }

    best = max(results, key=lambda k: results[k]["f1"])

    print(f"\n{'='*55}")
    print(f"  Best Model : {best}")
    print(f"  Best F1    : {results[best]['f1']:.4f}")
    print(f"  Run ID     : {results[best]['run_id']}")

    summary = {
        "best_model": str(best),
        "f1": float(results[best]["f1"]),
        "run_id": str(results[best]["run_id"]),
        "all_results": results
    }

    with open("best_model.json", "w") as f:
        json.dump(summary, f, indent=2)

    with mlflow.start_run(run_name="preprocessing_artifacts"):
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("best_model.json")

    print("\n✅ All baseline models trained and logged to MLflow!")
    print(f"   View at: {mlflow.get_tracking_uri()}")