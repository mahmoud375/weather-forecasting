"""
Centralized configuration for the Weather Forecasting project.

All constants — location coordinates, API parameters, file paths, and
feature definitions — live here so every module draws from a single
source of truth.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Project root (resolved relative to this file's location)
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────
# 1. Location
# ──────────────────────────────────────────────────────────────────────
LOCATION_NAME = os.getenv("WEATHER_LOCATION_NAME", "Cairo, Egypt")
LATITUDE = float(os.getenv("WEATHER_LATITUDE", "30.0444"))
LONGITUDE = float(os.getenv("WEATHER_LONGITUDE", "31.2357"))
TIMEZONE = os.getenv("WEATHER_TIMEZONE", "Africa/Cairo")

# ──────────────────────────────────────────────────────────────────────
# 2. API configuration — Open-Meteo Historical Weather API
# ──────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("OPEN_METEO_API_URL", "https://archive-api.open-meteo.com/v1/archive")

START_DATE = os.getenv("WEATHER_START_DATE", "2023-01-01")
END_DATE = os.getenv("WEATHER_END_DATE", "2025-12-31")

# Daily weather variables to request (7 features)
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "cloud_cover_mean",
]

# Human-readable column names for the cleaned dataset
COLUMN_RENAME_MAP = {
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "temperature_2m_mean": "temp_mean_c",
    "precipitation_sum": "precipitation_mm",
    "relative_humidity_2m_mean": "humidity_pct",
    "wind_speed_10m_max": "wind_speed_max_kmh",
    "cloud_cover_mean": "cloud_cover_pct",
}

# Units reference (for documentation / sanity checks)
UNITS = {
    "temp_max_c": "°C",
    "temp_min_c": "°C",
    "temp_mean_c": "°C",
    "precipitation_mm": "mm",
    "humidity_pct": "%",
    "wind_speed_max_kmh": "km/h",
    "cloud_cover_pct": "%",
}

# Physical plausibility bounds for Cairo
PHYSICAL_BOUNDS = {
    "temp_max_c": (-5, 55),
    "temp_min_c": (-10, 45),
    "temp_mean_c": (-5, 50),
    "precipitation_mm": (0, 200),     # Non-negative; extreme single-day cap
    "humidity_pct": (0, 100),
    "wind_speed_max_kmh": (0, 200),
    "cloud_cover_pct": (0, 100),
}

# ──────────────────────────────────────────────────────────────────────
# 3. File paths
# ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "artifacts", "figures")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

RAW_JSON_PATH = os.path.join(RAW_DATA_DIR, "open_meteo_response.json")
CLEAN_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "cairo_weather_clean.csv")
EXECUTIVE_SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "executive_summary.md")

# ──────────────────────────────────────────────────────────────────────
# 4. EDA parameters
# ──────────────────────────────────────────────────────────────────────
ROLLING_SHORT = 7       # 7-day rolling window
ROLLING_LONG = 30       # 30-day rolling window
ACF_LAGS = 60           # Lags for ACF/PACF
SEASONAL_PERIOD = 365   # Annual cycle for decomposition
ZSCORE_THRESHOLD = 3.0  # Outlier detection threshold
FIGURE_DPI = 150        # Publication-grade resolution


def ensure_dirs():
    """Create all output directories if they don't already exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
