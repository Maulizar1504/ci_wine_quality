"""
modelling_tuning.py
===================
Hyperparameter tuning using RandomizedSearchCV with full MLflow logging.
Trains best tuned models (RandomForest & GradientBoosting), logs all
tuning artifacts, and registers the best model.

Usage:
    1. Run modelling.py first (baseline)
    2. Start MLflow server:  mlflow server --host 127.0.0.1 --port 5000
    3. Run:                  python modelling_tuning.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv
import os

load_dotenv()

username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

import dagshub
import joblib
import matplotlib.pyplot as plt
import mlflow

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from wine_quality_preprocessing import load_data, preprocess

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()  # overridden by dagshub.init() below
EXPERIMENT_NAME     = "wine_quality_tuning"
N_ITER              = 30
CV_FOLDS            = 5
F1_THRESHOLD        = 0.70  # Quality gate

# ── DagsHub Integration (Advanced) ────────────────────────────
# Uncomment baris berikut untuk menyimpan artefak ke DagsHub:
dagshub.init(repo_owner='alsyamaulizar', repo_name='wine-quality-ml', mlflow=True)
# MLFLOW_TRACKING_URI akan di-override otomatis oleh dagshub.init()


def plot_cv_results(search, model_name: str) -> str:
    """Plot distribution of CV scores across all hyperparameter iterations."""
    scores = search.cv_results_["mean_test_score"]
    stds   = search.cv_results_["std_test_score"]
    sorted_idx = np.argsort(scores)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: sorted scores
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scores)))
    axes[0].bar(range(len(scores)), scores[sorted_idx],
                yerr=stds[sorted_idx], capsize=2, color=colors)
    axes[0].set_xlabel("Iteration (sorted by score)")
    axes[0].set_ylabel("Mean CV F1 Score")
    axes[0].set_title(f"Tuning Iterations — {model_name}", fontsize=12)
    axes[0].axhline(search.best_score_, color="red", linestyle="--",
                    label=f"Best: {search.best_score_:.4f}")
    axes[0].legend()

    # Histogram of scores
    axes[1].hist(scores, bins=15, color="steelblue", edgecolor="white")
    axes[1].axvline(search.best_score_, color="red", linestyle="--",
                    label=f"Best: {search.best_score_:.4f}")
    axes[1].set_xlabel("CV F1 Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Score Distribution — {model_name}", fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    path = f"tuning_cv_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Bad", "Good"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix (Tuned) — {model_name}", fontsize=13)
    path = f"cm_tuned_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc(model, X_test, y_test, model_name: str) -> str:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (Tuned) — {model_name}", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = f"roc_tuned_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(model, feature_names, model_name: str):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(imp)), imp[idx],
           color=plt.cm.viridis(np.linspace(0.2, 0.85, len(imp))))
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Feature Importances (Tuned) — {model_name}", fontsize=13)
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = f"fi_tuned_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    return path


def tune_and_log(
    model, param_grid: dict, model_name: str,
    X_trainval, y_trainval,
    X_test, y_test,
    feature_names: list,
) -> tuple:
    """Run RandomizedSearchCV and log everything to MLflow."""

    with mlflow.start_run(run_name=model_name) as run:
        print(f"\n  Running RandomizedSearchCV ({N_ITER} iters, {CV_FOLDS}-fold CV)…")

        cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=N_ITER,
    scoring=make_scorer(f1_score),
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=1,
    return_train_score=True,
)
        search.fit(X_trainval, y_trainval)
        best = search.best_estimator_

        # Evaluate on test set
        y_pred  = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_pred)
        test_f1  = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)

        # ── Log params & metrics ──────────────────────────────
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({
            "cv_best_f1"   : search.best_score_,
            "test_accuracy": test_acc,
            "test_f1"      : test_f1,
            "test_roc_auc" : test_auc,
            "n_iterations" : N_ITER,
            "cv_folds"     : CV_FOLDS,
        })
        mlflow.set_tags({
            "model_type"  : type(best).__name__,
            "tuning_method": "RandomizedSearchCV",
            "dataset"     : "wine_quality",
        })

        # ── Artifacts ─────────────────────────────────────────
        # 1. CV results plot
        cv_plot = plot_cv_results(search, model_name)
        mlflow.log_artifact(cv_plot)

        # 2. Confusion matrix
        cm_plot = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_plot)

        # 3. ROC Curve
        roc_plot = plot_roc(best, X_test, y_test, model_name)
        mlflow.log_artifact(roc_plot)

        # 4. Feature importance
        fi_plot = plot_feature_importance(best, feature_names, model_name)
        if fi_plot:
            mlflow.log_artifact(fi_plot)

        # 5. Best hyperparameters JSON
        with open(f"best_params_{model_name}.json", "w") as f:
            json.dump(search.best_params_, f, indent=2)
        mlflow.log_artifact(f"best_params_{model_name}.json")

        # 6. Full CV results CSV
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(f"cv_results_{model_name}.csv", index=False)
        mlflow.log_artifact(f"cv_results_{model_name}.csv")

        # 7. Classification report TXT
        report = classification_report(y_test, y_pred, target_names=["Bad", "Good"])
        with open(f"report_tuned_{model_name}.txt", "w") as f:
            f.write(f"Model: {model_name}\nBest Params: {search.best_params_}\n\n{report}")
        mlflow.log_artifact(f"report_tuned_{model_name}.txt")

        # 8. Register model
        mlflow.sklearn.log_model(
            best,
            artifact_path=model_name,
            registered_model_name=f"WineQuality_{model_name}",
        )

        print(f"\n  ✓ {model_name}")
        print(f"    Best CV F1 : {search.best_score_:.4f}")
        print(f"    Test F1    : {test_f1:.4f}")
        print(f"    Test AUC   : {test_auc:.4f}")
        print(f"    Run ID     : {run.info.run_id}")
        print(f"    Best Params: {search.best_params_}")

        return test_f1, test_auc, run.info.run_id, best


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("   Wine Quality — Hyperparameter Tuning (Advanced)")
    print("=" * 60)

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = preprocess(df)

    # Combine train + val for tuning (use full CV for validation)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)

    configs = {
        "RandomForest_Tuned": {
            "model" : RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators"     : [100, 200, 300, 500],
                "max_depth"        : [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf" : [1, 2, 4],
                "max_features"     : ["sqrt", "log2"],
                "bootstrap"        : [True, False],
            },
        },
        "GradientBoosting_Tuned": {
            "model" : GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators"     : [100, 200, 300, 500],
                "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
                "max_depth"        : [3, 5, 7, 9],
                "subsample"        : [0.7, 0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf" : [1, 2, 4],
            },
        },
    }

    all_results = {}
    best_f1     = 0.0
    best_model  = None
    best_name   = ""
    best_run_id = ""

    for name, cfg in configs.items():
        print(f"\n{'─'*60}")
        print(f"  Tuning: {name}")
        print(f"{'─'*60}")

        f1, auc, run_id, model = tune_and_log(
            cfg["model"], cfg["params"], name,
            X_trainval, y_trainval,
            X_test, y_test, features,
        )
        all_results[name] = {"test_f1": f1, "test_auc": auc, "run_id": run_id}

        if f1 > best_f1:
            best_f1     = f1
            best_model  = model
            best_name   = name
            best_run_id = run_id

    # ── Quality Gate ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Quality Gate Check: F1 >= {F1_THRESHOLD}")
    if best_f1 < F1_THRESHOLD:
        print(f"  ❌ FAILED: Best F1 {best_f1:.4f} < {F1_THRESHOLD}")
    else:
        print(f"  ✅ PASSED: Best F1 {best_f1:.4f} >= {F1_THRESHOLD}")

    print(f"\n  Best Tuned Model : {best_name}")
    print(f"  Best F1          : {best_f1:.4f}")
    print(f"  Best Run ID      : {best_run_id}")

    # ── Save best model locally ───────────────────────────────
    joblib.dump(best_model, "best_tuned_model.pkl")
    info = {
        "model_name": best_name,
        "test_f1"   : best_f1,
        "run_id"    : best_run_id,
        "all_results": all_results,
        "quality_gate_passed": best_f1 >= F1_THRESHOLD,
    }
    with open("best_tuned_info.json", "w") as f:
        json.dump(
    info,
    f,
    indent=2,
    default=lambda x:
        float(x) if isinstance(x, (np.floating, np.integer))
        else bool(x) if isinstance(x, np.bool_)
        else str(x)
)

    # Upload best model + scaler as artifact run
    with mlflow.start_run(run_name="best_tuned_artifacts"):
        mlflow.log_artifact("best_tuned_model.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("best_tuned_info.json")
        mlflow.log_metric("best_test_f1", best_f1)
        mlflow.set_tag("best_model", best_name)

    print("\n✅ Hyperparameter tuning complete!")
    print(f"   View at: {MLFLOW_TRACKING_URI}")
    print("   best_tuned_model.pkl saved.")
