#!/usr/bin/env python3
"""
train_models.py — Weather Forecasting Model Training & Serialization
===================================================================

Fits and exports production-grade machine learning models for the Django
web deployment (Task 6):
1. Primary Model: Random Forest Regressor (non-linear atmospheric dynamics)
2. Alternative Model: Ridge Linear Regression (interpretable baseline)

Both models predict daily mean temperature (temp_mean_c) from core meteorological
observations:
- temp_max_c (Maximum Temperature, °C)
- temp_min_c (Minimum Temperature, °C)
- humidity_pct (Relative Humidity, %)
- wind_speed_max_kmh (Maximum Wind Speed, km/h)
- precipitation_mm (Precipitation, mm)
- cloud_cover_pct (Cloud Cover, %)

Artifacts saved to:
- models/weather_model.joblib (Primary RF model)
- models/weather_model_ridge.joblib (Alternative Ridge model)
- models/model_metadata.json (Features, metrics, prediction intervals, bounds)
"""

import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cairo_weather_clean.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_models")

FEATURE_NAMES = [
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
    "wind_speed_max_kmh",
    "precipitation_mm",
    "cloud_cover_pct",
]
TARGET_NAME = "temp_mean_c"

FEATURE_BOUNDS = {
    "temp_max_c": {"min": -5.0, "max": 55.0, "unit": "°C", "label": "Max Temperature"},
    "temp_min_c": {"min": -5.0, "max": 55.0, "unit": "°C", "label": "Min Temperature"},
    "humidity_pct": {"min": 0.0, "max": 100.0, "unit": "%", "label": "Relative Humidity"},
    "wind_speed_max_kmh": {"min": 0.0, "max": 150.0, "unit": "km/h", "label": "Wind Speed"},
    "precipitation_mm": {"min": 0.0, "max": 300.0, "unit": "mm", "label": "Precipitation"},
    "cloud_cover_pct": {"min": 0.0, "max": 100.0, "unit": "%", "label": "Cloud Cover"},
}


def load_dataset() -> pd.DataFrame:
    """Load and validate the clean Cairo weather dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Clean weather data not found at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} daily weather records from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def train_and_export():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataset()

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # Chronological split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 1. Primary Model: Random Forest Regressor
    logger.info("Training Primary Model: RandomForestRegressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_r2 = float(r2_score(y_test, rf_preds))
    rf_residuals = y_test - rf_preds
    rf_residual_std = float(np.std(rf_residuals))

    logger.info(f"Random Forest  -> MAE: {rf_mae:.4f}°C | RMSE: {rf_rmse:.4f}°C | R²: {rf_r2:.4f} | Res Std: {rf_residual_std:.4f}°C")

    # 2. Alternative Model: Ridge Regression Pipeline
    logger.info("Training Alternative Model: Ridge Regression Pipeline...")
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    ridge_pipeline.fit(X_train, y_train)

    ridge_preds = ridge_pipeline.predict(X_test)
    ridge_mae = float(mean_absolute_error(y_test, ridge_preds))
    ridge_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_preds)))
    ridge_r2 = float(r2_score(y_test, ridge_preds))
    ridge_residuals = y_test - ridge_preds
    ridge_residual_std = float(np.std(ridge_residuals))

    logger.info(f"Ridge Pipeline -> MAE: {ridge_mae:.4f}°C | RMSE: {ridge_rmse:.4f}°C | R²: {ridge_r2:.4f} | Res Std: {ridge_residual_std:.4f}°C")

    # Save artifacts
    rf_path = os.path.join(MODELS_DIR, "weather_model.joblib")
    ridge_path = os.path.join(MODELS_DIR, "weather_model_ridge.joblib")

    joblib.dump(rf_model, rf_path)
    joblib.dump(ridge_pipeline, ridge_path)
    logger.info(f"Saved primary model to: {rf_path}")
    logger.info(f"Saved alternative model to: {ridge_path}")

    # Prepare metadata
    metadata = {
        "target": TARGET_NAME,
        "target_unit": "°C",
        "target_description": "Cairo Daily Mean Temperature",
        "features": FEATURE_NAMES,
        "feature_bounds": FEATURE_BOUNDS,
        "models": {
            "rf": {
                "name": "Random Forest Regressor (Primary)",
                "file": "weather_model.joblib",
                "mae": round(rf_mae, 4),
                "rmse": round(rf_rmse, 4),
                "r2": round(rf_r2, 4),
                "residual_std": round(rf_residual_std, 4),
                "ci_margin_95": round(1.96 * rf_residual_std, 4),
            },
            "ridge": {
                "name": "Ridge Linear Regression (Alternative)",
                "file": "weather_model_ridge.joblib",
                "mae": round(ridge_mae, 4),
                "rmse": round(ridge_rmse, 4),
                "r2": round(ridge_r2, 4),
                "residual_std": round(ridge_residual_std, 4),
                "ci_margin_95": round(1.96 * ridge_residual_std, 4),
            },
        },
        "dataset": {
            "samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "date_start": str(df["date"].min().date()),
            "date_end": str(df["date"].max().date()),
        },
        "default_cairo_example": {
            "temp_max_c": 30.5,
            "temp_min_c": 19.0,
            "humidity_pct": 52.0,
            "wind_speed_max_kmh": 16.5,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 25.0,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {METADATA_PATH}")

    # Verification sample prediction
    sample = pd.DataFrame([metadata["default_cairo_example"]])[FEATURE_NAMES]
    pred_rf = float(rf_model.predict(sample)[0])
    pred_ridge = float(ridge_pipeline.predict(sample)[0])
    logger.info("Sample Verification Prediction (Cairo typical day):")
    logger.info(f"  Inputs: {metadata['default_cairo_example']}")
    logger.info(f"  RF Prediction   : {pred_rf:.4f} °C (95% CI: [{pred_rf - 1.96*rf_residual_std:.2f}, {pred_rf + 1.96*rf_residual_std:.2f}])")
    logger.info(f"  Ridge Prediction: {pred_ridge:.4f} °C (95% CI: [{pred_ridge - 1.96*ridge_residual_std:.2f}, {pred_ridge + 1.96*ridge_residual_std:.2f}])")

    return metadata


if __name__ == "__main__":
    train_and_export()
