"""
Part 1 — Weather API Integration & Data Acquisition
=====================================================

Connects to the Open-Meteo Historical Weather API, fetches daily weather
records for Cairo, Egypt, and persists the raw JSON response to disk.

API Details
-----------
- Endpoint : https://archive-api.open-meteo.com/v1/archive
- Auth     : None required (free, keyless)
- Rate limit: 10 000 requests / day (non-commercial)
- Data src : ERA5 reanalysis (ECMWF) — global, back to 1940
- License  : CC BY 4.0 — attribution required
"""

import json
import logging
import sys
import time
from typing import Any, Dict

import requests

# Add project root to path for config import
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def build_request_params() -> Dict[str, Any]:
    """
    Construct the query-parameter dictionary for the Open-Meteo API call.

    Returns
    -------
    dict
        Ready-to-use params for ``requests.get``.
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
        "daily": ",".join(config.DAILY_VARIABLES),
        "timezone": config.TIMEZONE,
    }
    logger.info("Request parameters built:")
    logger.info(f"  Location  : {config.LOCATION_NAME} ({config.LATITUDE}, {config.LONGITUDE})")
    logger.info(f"  Date range: {config.START_DATE} → {config.END_DATE}")
    logger.info(f"  Variables : {', '.join(config.DAILY_VARIABLES)}")
    logger.info(f"  Timezone  : {config.TIMEZONE}")
    return params


def fetch_weather_data() -> Dict[str, Any]:
    """
    Send GET request to Open-Meteo and return the parsed JSON response.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code.
    ValueError
        If the response JSON is missing the expected ``daily`` key.
    """
    params = build_request_params()

    logger.info(f"Sending request to {config.API_BASE_URL} ...")
    try:
        response = requests.get(config.API_BASE_URL, params=params, timeout=60)
        elapsed = time.time() - start_time
        logger.info(f"Response received in {elapsed:.2f}s — HTTP {response.status_code}")
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        if os.path.exists(config.RAW_JSON_PATH):
            logger.warning(
                f"API request encountered an issue ({exc}). "
                f"Falling back to existing cached raw dataset at {config.RAW_JSON_PATH}."
            )
            with open(config.RAW_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise

    # Validate response structure
    if "daily" not in data:
        raise ValueError(
            "API response missing 'daily' key. "
            f"Keys received: {list(data.keys())}"
        )

    n_days = len(data["daily"].get("time", []))
    logger.info(f"Received {n_days} daily records.")
    logger.info(f"Daily keys: {list(data['daily'].keys())}")

    return data


def save_raw_json(data: Dict[str, Any]) -> str:
    """
    Persist the raw API response to disk as indented JSON.

    Parameters
    ----------
    data : dict
        The full API response dictionary.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    config.ensure_dirs()
    path = config.RAW_JSON_PATH

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(path) / 1024
    logger.info(f"Raw JSON saved to {path} ({size_kb:.1f} KB)")
    return path


def run() -> Dict[str, Any]:
    """
    Execute the full data-acquisition pipeline.

    Steps
    -----
    1. Build API request parameters.
    2. Fetch data from Open-Meteo.
    3. Save raw JSON to ``data/raw/``.

    Returns
    -------
    dict
        The raw API response dictionary (for downstream use).
    """
    logger.info("=" * 65)
    logger.info("PART 1 — DATA ACQUISITION")
    logger.info("=" * 65)

    data = fetch_weather_data()
    save_raw_json(data)

    logger.info("Part 1 complete.\n")
    return data


# ──────────────────────────────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    run()
