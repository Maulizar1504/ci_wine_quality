import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from wine_quality_preprocessing import load_data, preprocess

warnings.filterwarnings("ignore")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "wine_quality_tuning"


def plot_cv_results(search, nama_model):
    scores = search.cv_results_["mean_test_score"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted(scores, reverse=True), marker="o", linewidth=2)
    ax.axhline(search.best_score_, color="red", linestyle="--", label=f"Best: {search.best_score_:.4f}")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("CV F1 Score")
    ax.set_title(f"Hasil Tuning - {nama_model}")
    ax.legend()
    plt.tight_layout()
    path = f"tuning_{nama_model}.png"
    fig.savefig(path, dpi=120)
    plt.close()
    return path


def tuning_dan_log(model, param_grid, nama_model, X_trainval, y_trainval, X_test, y_test):
    with mlflow.start_run(run_name=nama_model) as run:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,
            scoring="f1",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_trainval, y_trainval)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrik = {
            "cv_f1": round(search.best_score_, 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrik)
        mlflow.set_tag("tuning_method", "RandomizedSearchCV")
        mlflow.log_artifact(plot_cv_results(search, nama_model))
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Buruk", "Bagus"], cmap="Blues", ax=ax
        )
        cm_path = f"cm_tuned_{nama_model}.png"
        fig.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)
        with open(f"best_params_{nama_model}.json", "w") as f:
            json.dump(search.best_params_, f, indent=2)
        mlflow.log_artifact(f"best_params_{nama_model}.json")
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(f"cv_results_{nama_model}.csv", index=False)
        mlflow.log_artifact(f"cv_results_{nama_model}.csv")
        mlflow.sklearn.log_model(best, artifact_path=nama_model)
        print(f"  CV F1: {metrik['cv_f1']} | Test F1: {metrik['test_f1']} | AUC: {metrik['test_roc_auc']}")
        return metrik["test_f1"], run.info.run_id, best


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    print("Memuat data...")
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess(df, save_scaler=False)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    print("Memulai tuning...")
    f1, run_id, best_model = tuning_dan_log(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, "RandomForest_Tuned",
        X_trainval, y_trainval, X_test, y_test,
    )
    joblib.dump(best_model, "best_model.pkl")
    info = {"model_name": "RandomForest_Tuned", "test_f1": f1, "run_id": run_id}
    with open("best_model.json", "w") as file:
        json.dump(info, file, indent=2)
    with mlflow.start_run(run_name="save_tuned_artifacts"):
        mlflow.log_artifact("best_model.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("best_model.json")
    print(f"\nTuning selesai! Best F1 = {f1}")
    print("best_model.pkl diperbarui!")
