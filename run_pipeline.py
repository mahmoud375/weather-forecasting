#!/usr/bin/env python3
"""
run_pipeline.py — Weather Forecasting Foundation Pipeline
==========================================================

Single entry point that executes the full Week 4 pipeline:
  Part 1 → Data Acquisition  (Open-Meteo API → raw JSON)
  Part 2 → Data Preparation  (raw JSON → clean CSV)
  Part 3 → Deep EDA          (clean CSV → figures + executive summary)

Usage
-----
    python run_pipeline.py

All outputs are written to:
    data/raw/              — Raw API response (JSON)
    data/processed/        — Cleaned dataset (CSV)
    artifacts/figures/     — EDA visualizations (PNG)
    artifacts/             — Executive summary (MD)
"""

import logging
import sys
import time

# Configure logging before importing modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")

# ──────────────────────────────────────────────────────────────────────
# Imports (after logging setup)
# ──────────────────────────────────────────────────────────────────────
import config
from src import data_acquisition, data_preparation, eda


def main():
    """Execute the full pipeline sequentially."""
    pipeline_start = time.time()

    logger.info("╔" + "═" * 63 + "╗")
    logger.info("║  WEATHER FORECASTING — FOUNDATION PIPELINE                    ║")
    logger.info("║  Location : Cairo, Egypt (30.04°N, 31.24°E)                   ║")
    logger.info("║  Period   : 2023-01-01 → 2025-12-31 (3 years, daily)          ║")
    logger.info("║  API      : Open-Meteo Historical Weather Archive             ║")
    logger.info("╚" + "═" * 63 + "╝")
    logger.info("")

    # Ensure output directories exist
    config.ensure_dirs()

    # ── Part 1: Data Acquisition ──────────────────────────────────────
    t1 = time.time()
    raw_data = data_acquisition.run()
    logger.info(f"  ⏱  Part 1 completed in {time.time() - t1:.2f}s\n")

    # ── Part 2: Data Preparation ──────────────────────────────────────
    t2 = time.time()
    df = data_preparation.run(raw_data)
    logger.info(f"  ⏱  Part 2 completed in {time.time() - t2:.2f}s\n")

    # ── Part 3: Deep EDA ──────────────────────────────────────────────
    t3 = time.time()
    eda_results = eda.run(df)
    logger.info(f"  ⏱  Part 3 completed in {time.time() - t3:.2f}s\n")

    # ── Summary ───────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    logger.info("╔" + "═" * 63 + "╗")
    logger.info("║  PIPELINE COMPLETE                                            ║")
    logger.info("╚" + "═" * 63 + "╝")
    logger.info(f"  Total time: {total:.2f}s")
    logger.info(f"  Raw data  : {config.RAW_JSON_PATH}")
    logger.info(f"  Clean CSV : {config.CLEAN_CSV_PATH}")
    logger.info(f"  Figures   : {config.FIGURES_DIR}/")
    logger.info(f"  Summary   : {config.EXECUTIVE_SUMMARY_PATH}")

    return eda_results


if __name__ == "__main__":
    main()
