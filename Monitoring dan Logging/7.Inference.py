"""
7.Inference.py
==============
Flask REST API untuk serving model Wine Quality Classification.
Dilengkapi dengan Prometheus metrics untuk monitoring.

Endpoints:
    POST /predict       — Inference satu atau beberapa sampel
    GET  /health        — Health check
    GET  /metrics       — Prometheus metrics scrape endpoint
    GET  /info          — Model metadata

Usage:
    1. Pastikan model sudah ditraining: python modelling_tuning.py
    2. Jalankan API:  python 7.Inference.py
    3. Test predict:
       curl -X POST http://localhost:5001/predict \\
            -H "Content-Type: application/json" \\
            -d '{"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]}'
"""

import logging
import os
import time
from datetime import datetime

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")

# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__)

# ── Prometheus Metrics ────────────────────────────────────────
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Inference request latency in seconds",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_COUNTER = Counter(
    "inference_predictions_total",
    "Total predictions by class",
    ["predicted_class"],
)
PREDICTION_CONFIDENCE = Histogram(
    "inference_prediction_confidence",
    "Prediction confidence score distribution",
    buckets=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)
ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Number of currently active inference requests",
)
MODEL_HEALTH = Gauge(
    "inference_model_loaded",
    "Whether the ML model is successfully loaded (1=yes, 0=no)",
)
TOTAL_ERRORS = Counter(
    "inference_errors_total",
    "Total inference errors by type",
    ["error_type"],
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "inference_batch_size",
    "Number of samples per inference request",
    buckets=[1, 2, 5, 10, 25, 50, 100],
)
MODEL_INFO_METRIC = Info("inference_model_info", "ML model metadata")

# ── Feature configuration ─────────────────────────────────────
ORIGINAL_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv(
    "MODEL_URI", "models:/WineQuality_GradientBoosting_Tuned/latest"
)
PORT = int(os.getenv("PORT", "5001"))

MODEL  = None
SCALER = None
START_TIME = time.time()


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as in preprocessing."""
    df = df.copy()
    df["total_acidity"]          = df["fixed acidity"] + df["volatile acidity"]
    df["free_sulfur_ratio"]      = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-8)
    df["alcohol_density_ratio"]  = df["alcohol"] / df["density"]
    df["acid_sugar_ratio"]       = df["citric acid"] / (df["residual sugar"] + 1e-8)
    df["sulfur_chloride_ratio"]  = df["total sulfur dioxide"] / (df["chlorides"] + 1e-8)
    return df


def load_model():
    """Load ML model and scaler."""
    global MODEL, SCALER
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        MODEL  = mlflow.sklearn.load_model(MODEL_URI)
        SCALER = joblib.load("scaler.pkl")
        MODEL_HEALTH.set(1)
        MODEL_INFO_METRIC.info({
            "model_uri"   : MODEL_URI,
            "framework"   : "scikit-learn",
            "task"        : "binary_classification",
            "classes"     : "bad,good",
            "loaded_at"   : datetime.now().isoformat(),
        })
        logger.info(f"Model loaded from MLflow: {MODEL_URI}")
    except Exception as e:
        logger.warning(f"MLflow load failed ({e}), trying local files…")
        try:
            MODEL  = joblib.load("best_tuned_model.pkl")
            SCALER = joblib.load("scaler.pkl")
            MODEL_HEALTH.set(1)
            logger.info("Model loaded from local best_tuned_model.pkl")
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
            MODEL_HEALTH.set(0)


# ── Routes ────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    ACTIVE_REQUESTS.inc()
    t0 = time.time()

    try:
        if MODEL is None or SCALER is None:
            TOTAL_ERRORS.labels(error_type="model_not_loaded").inc()
            return jsonify({"error": "Model not loaded"}), 503

        body = request.get_json(force=True, silent=True)
        if not body or "features" not in body:
            TOTAL_ERRORS.labels(error_type="invalid_input").inc()
            REQUEST_COUNT.labels(
                method="POST", endpoint="/predict", status_code="400"
            ).inc()
            return jsonify({
                "error": "Request body must contain 'features' key",
                "example": {
                    "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
                },
            }), 400

        raw = body["features"]

        # Accept: single list, list of lists, or dict
        if isinstance(raw, dict):
            df_in = pd.DataFrame([raw])
        elif isinstance(raw, list):
            if isinstance(raw[0], list):
                df_in = pd.DataFrame(raw, columns=ORIGINAL_FEATURES)
            else:
                df_in = pd.DataFrame([raw], columns=ORIGINAL_FEATURES)
        else:
            return jsonify({"error": "Invalid features format"}), 400

        if df_in.shape[1] != len(ORIGINAL_FEATURES):
            return jsonify({
                "error": f"Expected {len(ORIGINAL_FEATURES)} features, got {df_in.shape[1]}"
            }), 400

        # Feature engineering + scaling
        df_eng = apply_feature_engineering(df_in)
        X = SCALER.transform(df_eng)

        preds  = MODEL.predict(X)
        probas = MODEL.predict_proba(X)

        BATCH_SIZE_HISTOGRAM.observe(len(preds))

        predictions = []
        for pred, proba in zip(preds, probas):
            label      = "good" if pred == 1 else "bad"
            confidence = float(max(proba))
            PREDICTION_COUNTER.labels(predicted_class=label).inc()
            PREDICTION_CONFIDENCE.observe(confidence)
            predictions.append({
                "prediction"  : int(pred),
                "label"       : label,
                "confidence"  : round(confidence, 4),
                "probability" : {"bad": round(float(proba[0]), 4),
                                 "good": round(float(proba[1]), 4)},
            })

        latency = time.time() - t0
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(
            method="POST", endpoint="/predict", status_code="200"
        ).inc()

        return jsonify({
            "predictions"   : predictions,
            "num_samples"   : len(predictions),
            "latency_ms"    : round(latency * 1000, 2),
            "model_uri"     : MODEL_URI,
            "timestamp"     : datetime.now().isoformat(),
        })

    except Exception as exc:
        latency = time.time() - t0
        TOTAL_ERRORS.labels(error_type="prediction_error").inc()
        REQUEST_COUNT.labels(
            method="POST", endpoint="/predict", status_code="500"
        ).inc()
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500

    finally:
        ACTIVE_REQUESTS.dec()


@app.route("/health", methods=["GET"])
def health():
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code="200").inc()
    uptime = round(time.time() - START_TIME, 1)
    status = "healthy" if (MODEL is not None and SCALER is not None) else "degraded"
    return jsonify({
        "status"      : status,
        "model_loaded": MODEL is not None,
        "uptime_seconds": uptime,
        "timestamp"   : datetime.now().isoformat(),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/info", methods=["GET"])
def info():
    REQUEST_COUNT.labels(method="GET", endpoint="/info", status_code="200").inc()
    return jsonify({
        "model_uri"          : MODEL_URI,
        "framework"          : "scikit-learn",
        "task"               : "binary_classification",
        "classes"            : ["bad (0)", "good (1)"],
        "input_features"     : ORIGINAL_FEATURES,
        "num_features_input" : len(ORIGINAL_FEATURES),
        "num_features_total" : 16,  # after engineering
        "quality_threshold"  : 7,
        "version"            : "1.0.0",
    })


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading model…")
    load_model()
    logger.info(f"Starting inference server on port {PORT}")
    logger.info(f"Endpoints: /predict  /health  /metrics  /info")
    app.run(host="0.0.0.0", port=PORT, debug=False)
