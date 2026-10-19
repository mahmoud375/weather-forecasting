"""
Part 2 — Time-Series Dataset Preparation
==========================================

Reads the raw Open-Meteo JSON, constructs a clean pandas DataFrame with
proper datetime indexing, validates continuity and integrity, handles
missing values with domain-appropriate strategies, verifies units, and
exports the cleaned dataset as CSV.
"""

import json
import logging
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_raw_json() -> dict:
    """Load the raw API JSON from disk."""
    path = config.RAW_JSON_PATH
    logger.info(f"Loading raw JSON from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Parse the ``daily`` section of the Open-Meteo response into a
    pandas DataFrame with a DatetimeIndex.

    Parameters
    ----------
    data : dict
        Full API response.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``datetime`` index and one column per weather variable.
    """
    daily = data["daily"]

    df = pd.DataFrame(daily)

    # Parse 'time' → datetime and set as index
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "date"

    # Set explicit daily frequency
    df = df.asfreq("D")

    # Rename columns to clean names
    df = df.rename(columns=config.COLUMN_RENAME_MAP)

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} → {df.index.max()}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


# ──────────────────────────────────────────────────────────────────────
# Continuity & Integrity
# ──────────────────────────────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and remove duplicate timestamps.

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame.
    """
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"Found {n_dupes} duplicate timestamp(s) — dropping duplicates (keeping first).")
        df = df[~df.index.duplicated(keep="first")]
    else:
        logger.info("✓ No duplicate timestamps found.")
    return df


def check_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify date gaps by comparing the index against a complete daily
    date range.  Gaps are reported; ``asfreq('D')`` already inserted
    NaN rows for them.

    Returns
    -------
    pd.DataFrame
        The same DataFrame (gaps are already filled structurally by ``asfreq``).
    """
    expected = pd.date_range(start=config.START_DATE, end=config.END_DATE, freq="D")
    missing_dates = expected.difference(df.index)

    if len(missing_dates) > 0:
        logger.warning(
            f"Found {len(missing_dates)} missing date(s) in the timeline. "
            f"These will appear as NaN rows after asfreq('D')."
        )
        if len(missing_dates) <= 10:
            logger.warning(f"  Missing dates: {missing_dates.tolist()}")
    else:
        logger.info("✓ Timeline is continuous — no date gaps detected.")

    return df


# ──────────────────────────────────────────────────────────────────────
# Missing Values
# ──────────────────────────────────────────────────────────────────────

def report_missing(df: pd.DataFrame) -> pd.Series:
    """
    Quantify and log missing values per column.

    Returns
    -------
    pd.Series
        Count of NaNs per column.
    """
    missing = df.isna().sum()
    total = len(df)
    logger.info("Missing-value report:")
    for col, n in missing.items():
        pct = 100 * n / total
        status = f"  {col:30s}: {n:>4d} / {total} ({pct:.2f}%)"
        if n > 0:
            logger.warning(status)
        else:
            logger.info(status)
    return missing


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using domain-appropriate strategies.

    Strategy
    --------
    - **Temperature, humidity, wind speed, cloud cover:** Linear interpolation.
      These are physically smooth, continuous variables whose values change
      gradually day-to-day.  Interpolation respects the temporal structure
      better than forward-fill or mean imputation.

    - **Precipitation:** Fill with 0.0.
      In Cairo's desert climate, missing precipitation overwhelmingly
      corresponds to dry days.  Interpolation is inappropriate for
      precipitation because it is a sparse, event-driven variable
      (zero-inflated) — averaging between two rain events would create
      false rainfall on dry days.

    Returns
    -------
    pd.DataFrame
        DataFrame with no remaining NaNs.
    """
    # Separate strategies
    interp_cols = [
        "temp_max_c", "temp_min_c", "temp_mean_c",
        "humidity_pct", "wind_speed_max_kmh", "cloud_cover_pct",
    ]
    fill_zero_cols = ["precipitation_mm"]

    for col in interp_cols:
        if col in df.columns and df[col].isna().any():
            n_before = df[col].isna().sum()
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            logger.info(f"  {col}: interpolated {n_before} value(s) (linear)")

    for col in fill_zero_cols:
        if col in df.columns and df[col].isna().any():
            n_before = df[col].isna().sum()
            df[col] = df[col].fillna(0.0)
            logger.info(f"  {col}: filled {n_before} value(s) with 0.0 (desert climate assumption)")

    # Final check
    remaining = df.isna().sum().sum()
    if remaining > 0:
        logger.error(f"Still {remaining} NaN(s) remaining after imputation!")
    else:
        logger.info("✓ All missing values handled — zero NaNs remaining.")

    return df


# ──────────────────────────────────────────────────────────────────────
# Unit Verification & Sanity Bounds
# ──────────────────────────────────────────────────────────────────────

def verify_units_and_bounds(df: pd.DataFrame) -> None:
    """
    Confirm that all measurements are in expected units and within
    physically plausible ranges.

    Open-Meteo returns:
      - Temperature in °C
      - Precipitation in mm
      - Humidity in %
      - Wind speed in km/h
      - Cloud cover in %

    These match our target units, so no conversion is needed.
    """
    logger.info("Unit & bounds verification:")
    for col, (lo, hi) in config.PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        violations = ((df[col] < lo) | (df[col] > hi)).sum()
        unit = config.UNITS.get(col, "")
        if violations > 0:
            logger.warning(
                f"  ⚠ {col}: {violations} value(s) outside [{lo}, {hi}] {unit} "
                f"(actual range: {col_min:.2f}–{col_max:.2f})"
            )
        else:
            logger.info(
                f"  ✓ {col}: range [{col_min:.2f}, {col_max:.2f}] {unit} "
                f"— within bounds [{lo}, {hi}]"
            )


# ──────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────

def export_clean_csv(df: pd.DataFrame, use_cached: bool = None) -> str:
    """
    Save the validated, cleaned DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned time-series dataframe.
    use_cached : bool, optional
        If True and file exists, preserves the existing file. Defaults to config.USE_CACHED.

    Returns
    -------
    str
        Path to the exported CSV.
    """
    if use_cached is None:
        use_cached = config.USE_CACHED

    config.ensure_dirs()
    path = config.CLEAN_CSV_PATH

    if use_cached and os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        logger.info(
            f"Preserving existing clean CSV at {path} ({size_kb:.1f} KB, {len(df)} rows) "
            "(opted in via --use-cached / WEATHER_USE_CACHED)"
        )
        return path

    df.to_csv(path, float_format="%.2f")
    size_kb = os.path.getsize(path) / 1024
    logger.info(f"Clean dataset exported to {path} ({size_kb:.1f} KB, {len(df)} rows)")
    return path


# ──────────────────────────────────────────────────────────────────────
# Public helpers (for notebook / downstream use)
# ──────────────────────────────────────────────────────────────────────

def load_clean_data() -> pd.DataFrame:
    """
    Load the cleaned CSV dataset, parsing the date index.

    This is the primary entry point for downstream modules (EDA, modeling)
    that need the validated dataset.

    Returns
    -------
    pd.DataFrame
        The cleaned weather dataset with DatetimeIndex.
    """
    df = pd.read_csv(config.CLEAN_CSV_PATH, parse_dates=["date"], index_col="date")
    df = df.asfreq("D")
    return df


# ──────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────

def run(raw_data: dict = None, use_cached: bool = None) -> pd.DataFrame:
    """
    Execute the full data-preparation pipeline.

    Parameters
    ----------
    raw_data : dict, optional
        Pre-loaded API response dict.  If None, loads from disk.
    use_cached : bool, optional
        If True, reuse existing cleaned CSV if present and raw_data is None.
        Defaults to config.USE_CACHED (False).

    Returns
    -------
    pd.DataFrame
        The cleaned, validated DataFrame.
    """
    if use_cached is None:
        use_cached = config.USE_CACHED

    logger.info("=" * 65)
    logger.info("PART 2 — DATA PREPARATION")
    logger.info("=" * 65)

    # 1. Load
    if raw_data is None:
        raw_data = load_raw_json()

    # 2. Parse → DataFrame
    df = json_to_dataframe(raw_data)

    # 3. Continuity checks
    df = check_duplicates(df)
    df = check_gaps(df)

    # 4. Missing values
    report_missing(df)
    df = handle_missing(df)

    # 5. Unit & bounds verification
    verify_units_and_bounds(df)

    # 6. Export
    export_clean_csv(df, use_cached=use_cached)

    logger.info("Part 2 complete.\n")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    run()
