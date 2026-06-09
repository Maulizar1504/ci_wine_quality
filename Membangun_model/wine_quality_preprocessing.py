"""
Wine Quality Preprocessing Pipeline
Dataset: Red Wine Quality (UCI Machine Learning Repository)
Task: Binary Classification - Good (quality >= 7) vs Bad (quality < 7)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Constants ─────────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/wine-quality/winequality-red.csv"
)
QUALITY_THRESHOLD = 7
RANDOM_STATE = 42


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load Wine Quality dataset from local file or UCI repository."""
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath, sep=";")
        print(f"Data loaded from local file: {filepath}")
    else:
        df = pd.read_csv(DATA_URL, sep=";")
        print(f"Data loaded from URL: {DATA_URL}")

    print(f"Shape: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for better model performance."""
    df = df.copy()

    # Interaction features
    df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
    df["free_sulfur_ratio"] = (
        df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-8)
    )
    df["alcohol_density_ratio"] = df["alcohol"] / df["density"]
    df["acid_sugar_ratio"] = df["citric acid"] / (df["residual sugar"] + 1e-8)
    df["sulfur_chloride_ratio"] = (
        df["total sulfur dioxide"] / (df["chlorides"] + 1e-8)
    )

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary classification target."""
    df = df.copy()
    df["quality_label"] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)
    print(
        f"\nTarget distribution:\n"
        f"  Good (>=7): {df['quality_label'].sum()} "
        f"({df['quality_label'].mean()*100:.1f}%)\n"
        f"  Bad  (<7) : {(~df['quality_label'].astype(bool)).sum()} "
        f"({(1-df['quality_label'].mean())*100:.1f}%)"
    )
    return df


def preprocess(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    save_scaler: bool = True,
    output_dir: str = ".",
):
    """
    Full preprocessing pipeline.

    Returns:
        X_train, X_val, X_test (scaled numpy arrays)
        y_train, y_val, y_test (pandas Series)
        scaler (fitted StandardScaler)
        feature_cols (list of feature names)
    """
    df = feature_engineering(df)
    df = create_target(df)

    feature_cols = [c for c in df.columns if c not in ["quality", "quality_label"]]
    X = df[feature_cols].copy()
    y = df["quality_label"].copy()

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print(f"Filling {X.isnull().sum().sum()} missing values with median")
        X = X.fillna(X.median())

    # Split: 64% train | 16% val | 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
    )

    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"\nScaler saved → {scaler_path}")

    print(
        f"\nSplit sizes:"
        f"\n  Train : {X_train_scaled.shape[0]} samples"
        f"\n  Val   : {X_val_scaled.shape[0]} samples"
        f"\n  Test  : {X_test_scaled.shape[0]} samples"
        f"\n  Features: {len(feature_cols)}"
    )

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        scaler, feature_cols,
    )


if __name__ == "__main__":
    print("=" * 55)
    print("   Wine Quality Preprocessing Pipeline")
    print("=" * 55)

    df = load_data()

    print("\n--- Basic Info ---")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nDescriptive stats:\n{df.describe().round(2)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    print("\n--- Running Preprocessing ---")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = preprocess(df)

    print(f"\nFeatures used ({len(features)}):\n  {features}")
    print("\nPreprocessing completed successfully!")
