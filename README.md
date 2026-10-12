# Weather Forecasting — Foundation Phase (Week 4)

> [!IMPORTANT]
> ### 📌 Primary Deliverables & Key Artifacts
> - **Executive Summary Report:** [`artifacts/executive_summary.md`](artifacts/executive_summary.md) — Concise analytical report covering climate context, API parameters, ADF stationarity diagnostics, seasonal dynamics, and modeling guidance.
> - **Visual EDA Gallery:** [`artifacts/figures/`](artifacts/figures/) — 9 publication-grade visualizations (150 DPI) spanning time-series dynamics, rolling windows, STL decomposition, ACF/PACF, correlation, and rolling Z-score anomaly detection.
> - **Preprocessed Time-Series Dataset:** [`data/processed/cairo_weather_clean.csv`](data/processed/cairo_weather_clean.csv) — 1,096 continuous daily observations (2023–2025) across 7 normalized meteorological variables (zero missing values, validated bounds).
> - **Interactive Evaluation Notebook:** [`notebooks/main_pipeline.ipynb`](notebooks/main_pipeline.ipynb) — Executable Jupyter notebook providing step-by-step walkthrough of Parts 1, 2, and 3 with inline plots and analysis.

---

End-to-end weather forecasting ML system: data acquisition, time-series structuring, and deep exploratory data analysis.

## 📍 Project Scope

| Detail          | Value                                       |
|----------------|---------------------------------------------|
| **Location**   | Cairo, Egypt (30.0444°N, 31.2357°E)         |
| **Period**     | 2023-01-01 → 2025-12-31 (3 years, daily)    |
| **API**        | Open-Meteo Historical Weather Archive       |
| **Variables**  | 7 daily features (temp, precip, humidity, wind, cloud cover) |

## 🚀 Quickstart & Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic, and modern Python package and environment management.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Linux/macOS (.venv\Scripts\activate on Windows)
uv pip install -r requirements.txt

# 3. (Optional) Configure environment overrides
cp .env.example .env

# 4. Run the full pipeline (Acquisition → Preparation → EDA)
uv run run_pipeline.py
```

## 📁 Project Structure

```
weather-forecasting/
├── config.py                          # Centralized configuration (location, paths, bounds)
├── run_pipeline.py                    # Single orchestration entry point
├── requirements.txt                   # Project dependencies (pinned minimums)
├── .env.example                       # Template for environment configuration overrides
├── .env                               # Local environment overrides
├── .gitignore                         # Git exclusion rules (.venv, pycache, OS files)
├── README.md                          # Project documentation and quickstart
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py            # Part 1: API fetch & raw JSON persistence
│   ├── data_preparation.py            # Part 2: Parsing, cleaning, validation & CSV export
│   └── eda.py                         # Part 3: Deep time-series EDA & statistical tests
├── data/
│   ├── raw/
│   │   └── open_meteo_response.json   # Raw API response payload (107 KB)
│   └── processed/
│       └── cairo_weather_clean.csv    # Validated, cleaned dataset (1,096 rows × 7 features)
├── artifacts/
│   ├── figures/                       # Publication-grade figures (PNG, 150 DPI)
│   │   ├── 01_distributions.png
│   │   ├── 02_temporal_lines.png
│   │   ├── 03_rolling_means.png
│   │   ├── 04_monthly_boxplots.png
│   │   ├── 05_volatility.png
│   │   ├── 06_seasonal_decomposition.png
│   │   ├── 07_acf_pacf.png
│   │   ├── 08_correlation_heatmap.png
│   │   └── 09_rolling_zscore_outliers.png
│   └── executive_summary.md           # ~0.5-page executive summary answering required questions
└── notebooks/
    └── main_pipeline.ipynb            # Interactive walkthrough notebook for evaluation
```

## 📊 Features Collected

| Variable         | API Parameter                | Unit  | Description |
|------------------|------------------------------|-------|-------------|
| Max Temperature  | `temperature_2m_max`         | °C    | Maximum daily air temperature at 2m |
| Min Temperature  | `temperature_2m_min`         | °C    | Minimum daily air temperature at 2m |
| Mean Temperature | `temperature_2m_mean`        | °C    | Mean daily air temperature at 2m |
| Precipitation    | `precipitation_sum`          | mm    | Daily precipitation sum |
| Humidity         | `relative_humidity_2m_mean`  | %     | Mean relative humidity at 2m |
| Wind Speed       | `wind_speed_10m_max`         | km/h  | Maximum wind speed at 10m |
| Cloud Cover      | `cloud_cover_mean`           | %     | Mean cloud cover percentage |

## 📈 EDA Visualizations & Statistical Tests

The pipeline generates 9 publication-grade figures under `artifacts/figures/`:
1. **Feature Distributions** (`01_distributions.png`) — Histograms with kernel density estimates (KDE) across all variables.
2. **Temporal Dynamics** (`02_temporal_lines.png`) — Full 3-year timeline for daily temperatures (min, max, mean envelope) and precipitation bars.
3. **Rolling Trend Analysis** (`03_rolling_means.png`) — 7-day and 30-day moving averages smoothing seasonal trajectories.
4. **Monthly Seasonality** (`04_monthly_boxplots.png`) — Monthly distributions illustrating seasonal cycles and intra-month dispersion.
5. **Temperature Volatility** (`05_volatility.png`) — 30-day rolling standard deviations identifying seasonal transition variance.
6. **Seasonal Decomposition** (`06_seasonal_decomposition.png`) — Classical additive decomposition into Trend, Annual Seasonal (period=365), and Residual.
7. **Autocorrelation & Partial Autocorrelation** (`07_acf_pacf.png`) — ACF and PACF plots up to 60 lags diagnosing persistence and autoregressive structure.
8. **Correlation Matrix** (`08_correlation_heatmap.png`) — Triangular Pearson correlation matrix evaluating inter-variable colinearity.
9. **Contextual Outlier Detection** (`09_rolling_zscore_outliers.png`) — Rolling 30-day Z-scores (|Z| > 3.0) isolating true anomalies from seasonal temperature shifts.

### Statistical Stationarity (ADF Test)
- **Mean Temperature:** ADF statistic = `-2.0139`, p-value = `0.2805` &rarr; **Non-stationary** (unit root present; seasonal differencing or detrending required).
- **Precipitation:** ADF statistic = `-30.5373`, p-value = `0.0000` &rarr; **Stationary** (zero-inflated point process).

## 🔧 API Details

- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Authentication:** None required (free, keyless)
- **Rate Limit:** 10,000 requests/day
- **Data Source:** ECMWF ERA5 Reanalysis
- **License:** CC BY 4.0

## 📜 Attribution

Weather data provided by [Open-Meteo](https://open-meteo.com/) under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
