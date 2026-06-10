import json
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from wine_quality_preprocessing import load_data, preprocess

warnings.filterwarnings("ignore")

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "wine_quality_baseline"


def simpan_confusion_matrix(y_true, y_pred, nama_model):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Buruk", "Bagus"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix - {nama_model}")
    path = f"cm_{nama_model}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def simpan_roc_curve(model, X_test, y_test, nama_model):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {nama_model}")
    ax.legend()
    plt.tight_layout()
    path = f"roc_{nama_model}.png"
    fig.savefig(path, dpi=120)
    plt.close()
    return path


def latih_dan_log(model, nama_model, X_train, X_val, X_test, y_train, y_val, y_test):
    with mlflow.start_run(run_name=nama_model) as run:
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        metrik = {
            "val_accuracy": round(accuracy_score(y_val, y_pred_val), 4),
            "val_f1": round(f1_score(y_val, y_pred_val, zero_division=0), 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred_test), 4),
            "test_f1": round(f1_score(y_test, y_pred_test, zero_division=0), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_proba_test), 4),
        }
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrik)
        mlflow.set_tag("model", nama_model)
        mlflow.log_artifact(simpan_confusion_matrix(y_test, y_pred_test, nama_model))
        mlflow.log_artifact(simpan_roc_curve(model, X_test, y_test, nama_model))
        laporan = classification_report(y_test, y_pred_test, target_names=["Buruk", "Bagus"])
        with open(f"report_{nama_model}.txt", "w") as f:
            f.write(laporan)
        mlflow.log_artifact(f"report_{nama_model}.txt")
        mlflow.sklearn.log_model(model, artifact_path=nama_model)
        print(f"  {nama_model} | val_f1={metrik['val_f1']} | test_f1={metrik['test_f1']}")
        return metrik["test_f1"], run.info.run_id, model


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    print("Memuat data...")
    df = load_data()
# Tangkap 6 variabel hasil split data saja
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

# Karena scaler sudah disimpan otomatis oleh preprocess ke './scaler.pkl', 
# jika code di bawah baris ini membutuhkan variabel `scaler`, load manual seperti ini:
import joblib
scaler = joblib.load("./scaler.pkl")
    daftar_model = {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    print("\nHasil pelatihan:")
    hasil = {}
    for nama, model in daftar_model.items():
        f1, run_id, trained_model = latih_dan_log(
            model, nama, X_train, X_val, X_test, y_train, y_val, y_test
        )
        hasil[nama] = {"f1": f1, "run_id": run_id, "model": trained_model}
    terbaik = max(hasil, key=lambda k: hasil[k]["f1"])
    joblib.dump(hasil[terbaik]["model"], "best_model.pkl")
    info = {"model_name": terbaik, "test_f1": hasil[terbaik]["f1"]}
    with open("best_model.json", "w") as f:
        json.dump(info, f, indent=2)
    with mlflow.start_run(run_name="save_artifacts"):
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("best_model.pkl")
        mlflow.log_artifact("best_model.json")
    print(f"\nModel terbaik: {terbaik} | F1={hasil[terbaik]['f1']}")
    print("best_model.pkl tersimpan!")
