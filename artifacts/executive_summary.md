# Executive Summary — Cairo Weather Time-Series Analysis

## Region & Time Window
**Location:** Cairo, Egypt (30.0444°N, 31.2357°E) — hot desert climate (Köppen: BWh).
**Period:** January 1, 2023 – December 31, 2025 (3 full years, 1096 daily records).
**Split Sizes:** Planned training set (2023–2024) = 731 rows (accounting for the 2024 leap year); test set (2025) = 365 rows.
**Rationale:** Cairo offers a clear seasonal temperature cycle (~16°C winters → ~32°C summers) and extremely sparse precipitation, providing a clean testbed for time-series decomposition and forecasting. Three years enables robust seasonal pattern detection and meaningful train/val/test splitting.

## API & Data Collection
**API:** Open-Meteo Historical Weather API (`archive-api.open-meteo.com`) — free, keyless, backed by ECMWF ERA5 reanalysis.
**Variables:** 7 daily features (temperature min/max/mean, precipitation, humidity, wind speed, cloud cover) exported in cleaned, unscaled variables in physical units.
**Rate limits:** 10,000 requests/day; our single-call fetch used 1 request. No throttling or pagination needed.
**Raw response** preserved as immutable JSON for full reproducibility.

## Key Temporal Patterns
- **Annual temperature cycle:** Mean temperature ranges from 10.3°C to 36.1°C (with overall extremes from 4.6°C daily min to 46.4°C daily max), exhibiting a clear sinusoidal seasonal pattern (winter avg ~15.9°C, summer avg ~31.6°C).
- **Temperature volatility** is highest during seasonal transitions (spring/autumn equinoxes), with lower day-to-day variance in mid-summer and mid-winter.
- **Precipitation** is extremely sparse: 66 days with measurable rain out of 1096 (94.0% dry days), totaling ~52.6 mm over 3 years.
- **Rolling-window trend analysis** reveals consistent year-over-year seasonality with no pronounced multi-year warming or cooling trend in this short window.

## Stationarity Assessment
- **Mean Temperature:**
  - **Raw Series (N=1096):** ADF statistic = -2.0139 (p = 0.2805); KPSS statistic = 0.2351 (p = 0.1000). ADF fails to reject while KPSS does not reject stationarity. The evidence is inconclusive and seasonality-driven rather than a clear unit root.
  - **First Difference (d=1, N=1095):** ADF statistic = -15.1740 (p = 6.331975e-28); KPSS statistic = 0.1501 (p = 0.1000) → Confirmed stationary.
  - **Seasonal Difference (D=1, s=365, leaves 731 rows):** ADF statistic = -9.4576 (p = 4.442423e-16); KPSS statistic = 0.3906 (p = 0.0812) → Confirmed stationary.
  - **Modeling Guidance:** We recommend first differencing or Fourier/calendar cyclic features over seasonal differencing at lag 365, which incurs a steep penalty of 365 rows lost (one full year of training data).
- **Precipitation (N=1096):** ADF test statistic = -30.5373, p-value = 0.000000e+00 → **Stationary** (point process dominated by 94.0% dry days). Modeling guidance: Standard regression with MSE loss suffers from severe zero-inflation bias. A two-stage hurdle model (classification for rain occurrence + regression for conditional rain volume) or Tweedie loss is recommended.

## Data Quality & Anomalies
- **Completeness:** No gaps or duplicate timestamps were found; the timeline is strictly continuous with explicit `<Day>` frequency.
- **Missing values:** The raw API response contained 0 nulls across all features; the linear interpolation and zero-fill logic in the pipeline is defensive and was not exercised on this dataset. Final dataset has zero NaNs.
- **Outliers:** 1 temperature outlier(s) detected via rolling 30-day Z-score (|Z| > 3.0). On 2025-04-22, mean temperature reached 34.1°C (Z = +3.05). This rapid spring thermal spike is consistent with a Khamsin-type event; not externally verified.
- **Sanity checks passed:** No negative precipitation, no temperature inversions (min > max), all percentages within [0, 100]%.
