"""
ml_service.py — Dedicated Machine Learning Inference Service
============================================================

Implements a thread-safe Singleton pattern to ensure ML models are loaded
EXACTLY ONCE when Django initializes (via AppConfig.ready). Model weights
are kept resident in memory and never re-read from disk during HTTP requests.

Includes:
- Thread-safe singleton instance management
- In-memory LRU caching of predictions for sub-millisecond repeated responses
- 95% confidence prediction intervals derived from empirical residual variance
- Dynamic weather condition categorization and icon badges
"""

from functools import lru_cache
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("forecast.ml_service")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


class WeatherModelService:
    """
    Thread-safe Singleton ML Inference Service for Cairo Weather Forecasting.
    """
    _instance: Optional["WeatherModelService"] = None
    _lock = threading.Lock()

    def __init__(self):
        if WeatherModelService._instance is not None:
            raise RuntimeError("WeatherModelService is a singleton. Use get_instance().")
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._is_loaded = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "WeatherModelService":
        """Get or initialize the thread-safe singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_models(self) -> None:
        """
        Load serialized models and metadata into memory once.
        Called during AppConfig.ready(). Safe to call repeatedly.
        """
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            logger.info("Initializing WeatherModelService: loading model artifacts into memory...")
            
            # Load metadata
            if not os.path.exists(METADATA_PATH):
                raise FileNotFoundError(f"Model metadata not found at {METADATA_PATH}. Run src/train_models.py first.")
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            # Load models
            rf_path = os.path.join(MODELS_DIR, "weather_model.joblib")
            ridge_path = os.path.join(MODELS_DIR, "weather_model_ridge.joblib")

            if not os.path.exists(rf_path):
                raise FileNotFoundError(f"Primary model not found at {rf_path}")
            if not os.path.exists(ridge_path):
                raise FileNotFoundError(f"Alternative model not found at {ridge_path}")

            self._models["rf"] = joblib.load(rf_path)
            self._models["ridge"] = joblib.load(ridge_path)
            self._is_loaded = True

            logger.info("WeatherModelService successfully loaded Random Forest and Ridge models.")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def metadata(self) -> Dict[str, Any]:
        if not self._is_loaded:
            self.load_models()
        return self._metadata

    @property
    def feature_names(self) -> List[str]:
        return self.metadata.get("features", [
            "temp_max_c",
            "temp_min_c",
            "humidity_pct",
            "wind_speed_max_kmh",
            "precipitation_mm",
            "cloud_cover_pct",
        ])

    def get_model(self, model_key: str = "rf") -> Any:
        """Retrieve model object by key ('rf' or 'ridge')."""
        if not self._is_loaded:
            self.load_models()
        if model_key not in self._models:
            model_key = "rf"
        return self._models[model_key]

    @staticmethod
    def classify_weather(
        precipitation_mm: float,
        cloud_cover_pct: float,
        wind_speed_max_kmh: float,
        temp_max_c: float,
    ) -> Dict[str, str]:
        """
        Derive intuitive weather condition, icon, and contextual badge from inputs.
        """
        if precipitation_mm > 1.0:
            return {
                "condition": "Rainy",
                "icon": "bi-cloud-rain-heavy-fill",
                "color": "primary",
                "bg_gradient": "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)",
                "description": f"Precipitation expected: {precipitation_mm:.1f} mm",
            }
        elif wind_speed_max_kmh >= 35.0:
            return {
                "condition": "Windy / Khamsin",
                "icon": "bi-wind",
                "color": "warning",
                "bg_gradient": "linear-gradient(135deg, #f39c12 0%, #d35400 100%)",
                "description": f"High winds detected: {wind_speed_max_kmh:.1f} km/h",
            }
        elif cloud_cover_pct >= 75.0:
            return {
                "condition": "Overcast",
                "icon": "bi-clouds-fill",
                "color": "secondary",
                "bg_gradient": "linear-gradient(135deg, #606c88 0%, #3f4c6b 100%)",
                "description": f"Dense cloud cover: {cloud_cover_pct:.0f}%",
            }
        elif cloud_cover_pct >= 30.0:
            return {
                "condition": "Partly Cloudy",
                "icon": "bi-cloud-sun-fill",
                "color": "info",
                "bg_gradient": "linear-gradient(135deg, #3a7bd5 0%, #3a6073 100%)",
                "description": f"Scattered clouds: {cloud_cover_pct:.0f}%",
            }
        elif temp_max_c >= 35.0:
            return {
                "condition": "Hot & Sunny",
                "icon": "bi-sun-fill",
                "color": "danger",
                "bg_gradient": "linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)",
                "description": f"High temperature: {temp_max_c:.1f} °C",
            }
        else:
            return {
                "condition": "Clear & Sunny",
                "icon": "bi-brightness-high-fill",
                "color": "warning",
                "bg_gradient": "linear-gradient(135deg, #f12711 0%, #f5af19 100%)",
                "description": "Clear skies across the Cairo basin",
            }

    def predict(
        self,
        features: Dict[str, float],
        model_key: str = "rf",
    ) -> Dict[str, Any]:
        """
        Run inference with in-memory caching and interval calculation.
        """
        if not self._is_loaded:
            self.load_models()

        if model_key not in ("rf", "ridge"):
            model_key = "rf"

        # Canonical tuple for caching
        feature_tuple = tuple(float(features[name]) for name in self.feature_names)
        
        # Call cached core inference
        pred_value, is_cached = self._predict_cached(model_key, feature_tuple)

        # Retrieve model metadata for prediction intervals
        model_meta = self.metadata["models"].get(model_key, self.metadata["models"]["rf"])
        ci_margin = float(model_meta.get("ci_margin_95", 0.85))

        weather_info = self.classify_weather(
            precipitation_mm=features.get("precipitation_mm", 0.0),
            cloud_cover_pct=features.get("cloud_cover_pct", 0.0),
            wind_speed_max_kmh=features.get("wind_speed_max_kmh", 0.0),
            temp_max_c=features.get("temp_max_c", 25.0),
        )

        return {
            "prediction": round(pred_value, 2),
            "prediction_exact": pred_value,
            "target": self.metadata.get("target", "temp_mean_c"),
            "units": self.metadata.get("target_unit", "°C"),
            "model_key": model_key,
            "model_used": model_meta["name"],
            "prediction_interval": {
                "lower": round(pred_value - ci_margin, 2),
                "upper": round(pred_value + ci_margin, 2),
                "margin": round(ci_margin, 2),
                "confidence": "95%",
            },
            "weather_condition": weather_info,
            "is_cached": is_cached,
            "metrics": {
                "mae": model_meta.get("mae"),
                "rmse": model_meta.get("rmse"),
                "r2": model_meta.get("r2"),
            },
        }

    def _predict_cached(self, model_key: str, feature_tuple: Tuple[float, ...]) -> Tuple[float, bool]:
        """Wrapper to track cache hits vs misses."""
        cache_key = (model_key, feature_tuple)
        info_before = _raw_predict_memoized.cache_info()
        pred = _raw_predict_memoized(self, model_key, feature_tuple)
        info_after = _raw_predict_memoized.cache_info()
        was_hit = info_after.hits > info_before.hits
        return pred, was_hit


@lru_cache(maxsize=1024)
def _raw_predict_memoized(service: WeatherModelService, model_key: str, feature_tuple: Tuple[float, ...]) -> float:
    """Core pure function decorated with lru_cache."""
    model = service.get_model(model_key)
    names = service.feature_names
    df_input = pd.DataFrame([dict(zip(names, feature_tuple))])[names]
    pred = float(model.predict(df_input)[0])
    return pred
