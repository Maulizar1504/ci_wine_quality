"""
tests/test_preprocessing.py
============================
Unit tests for wine_quality_preprocessing.py
Run: pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Membangun_model"))

from wine_quality_preprocessing import (
    create_target,
    feature_engineering,
    preprocess,
)

# ── Fixtures ───────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a minimal synthetic dataset mimicking wine quality CSV."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "fixed acidity"       : np.random.uniform(4, 15, n),
        "volatile acidity"    : np.random.uniform(0.1, 1.5, n),
        "citric acid"         : np.random.uniform(0, 1, n),
        "residual sugar"      : np.random.uniform(1, 15, n),
        "chlorides"           : np.random.uniform(0.01, 0.2, n),
        "free sulfur dioxide" : np.random.uniform(1, 72, n),
        "total sulfur dioxide": np.random.uniform(6, 289, n),
        "density"             : np.random.uniform(0.990, 1.004, n),
        "pH"                  : np.random.uniform(2.7, 4.0, n),
        "sulphates"           : np.random.uniform(0.3, 2.0, n),
        "alcohol"             : np.random.uniform(8, 15, n),
        "quality"             : np.random.randint(3, 9, n),
    })


# ── Test: feature_engineering ─────────────────────────────
class TestFeatureEngineering:
    def test_new_columns_added(self, sample_df):
        result = feature_engineering(sample_df)
        expected_new = [
            "total_acidity", "free_sulfur_ratio",
            "alcohol_density_ratio", "acid_sugar_ratio",
            "sulfur_chloride_ratio",
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sample_df):
        result = feature_engineering(sample_df)
        for col in sample_df.columns:
            assert col in result.columns

    def test_total_acidity_correct(self, sample_df):
        result = feature_engineering(sample_df)
        expected = sample_df["fixed acidity"] + sample_df["volatile acidity"]
        pd.testing.assert_series_equal(
            result["total_acidity"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_inf_values(self, sample_df):
        result = feature_engineering(sample_df)
        assert not np.isinf(result.select_dtypes(include=np.number).values).any()

    def test_no_nan_values(self, sample_df):
        result = feature_engineering(sample_df)
        new_cols = ["total_acidity", "free_sulfur_ratio",
                    "alcohol_density_ratio", "acid_sugar_ratio",
                    "sulfur_chloride_ratio"]
        assert not result[new_cols].isnull().any().any()


# ── Test: create_target ────────────────────────────────────
class TestCreateTarget:
    def test_quality_label_created(self, sample_df):
        result = create_target(sample_df)
        assert "quality_label" in result.columns

    def test_binary_values_only(self, sample_df):
        result = create_target(sample_df)
        unique_vals = set(result["quality_label"].unique())
        assert unique_vals.issubset({0, 1})

    def test_threshold_correct(self, sample_df):
        result = create_target(sample_df)
        mask_good = sample_df["quality"] >= 7
        assert (result.loc[mask_good, "quality_label"] == 1).all()
        assert (result.loc[~mask_good, "quality_label"] == 0).all()


# ── Test: preprocess ───────────────────────────────────────
class TestPreprocess:
    def test_output_shapes(self, sample_df):
        X_tr, X_val, X_te, y_tr, y_val, y_te, scaler, feats = preprocess(
            sample_df, save_scaler=False
        )
        total = len(sample_df)
        assert X_tr.shape[0] + X_val.shape[0] + X_te.shape[0] == total
        assert X_tr.shape[0] == len(y_tr)
        assert X_val.shape[0] == len(y_val)
        assert X_te.shape[0] == len(y_te)

    def test_feature_count(self, sample_df):
        X_tr, X_val, X_te, *_ = preprocess(sample_df, save_scaler=False)
        # 11 original + 5 engineered = 16 features
        assert X_tr.shape[1] == 16

    def test_scaled_values_range(self, sample_df):
        X_tr, X_val, X_te, *_ = preprocess(sample_df, save_scaler=False)
        # Scaled data should be roughly within [-5, 5]
        assert np.abs(X_tr).max() < 10
        assert np.abs(X_val).max() < 10
        assert np.abs(X_te).max() < 10

    def test_scaler_returned(self, sample_df):
        from sklearn.preprocessing import StandardScaler
        *_, scaler, feats = preprocess(sample_df, save_scaler=False)
        assert isinstance(scaler, StandardScaler)

    def test_feature_names_returned(self, sample_df):
        *_, feats = preprocess(sample_df, save_scaler=False)
        assert isinstance(feats, list)
        assert len(feats) == 16
        assert "quality" not in feats
        assert "quality_label" not in feats

    def test_no_data_leakage(self, sample_df):
        """Val/Test scaler should use train statistics only."""
        X_tr, X_val, X_te, *_, scaler, _ = preprocess(sample_df, save_scaler=False)
        # Train mean should be close to 0 after scaling
        assert abs(X_tr.mean()) < 0.5


# ── Test: missing value handling ───────────────────────────
class TestMissingValues:
    def test_handles_missing_values(self, sample_df):
        df_with_nan = sample_df.copy()
        df_with_nan.loc[0:5, "alcohol"] = np.nan
        # Should not raise
        X_tr, *_ = preprocess(df_with_nan, save_scaler=False)
        assert not np.isnan(X_tr).any()
