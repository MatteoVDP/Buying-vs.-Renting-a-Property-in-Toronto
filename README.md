# Buying vs. Renting a Property in Toronto

This repository contains a multi-model forecasting system for Toronto detached-home price dynamics, designed to support long-horizon buy-vs-rent scenario analysis.

The project combines macroeconomic simulation, feature-engineered machine learning, Monte Carlo forecasting, and post-processing scripts for diagnostics and stress testing.

## Project Goals

- Forecast housing market trajectories over long horizons (typically 25 years / 300 months).
- Capture uncertainty with Monte Carlo simulation rather than single-point predictions.
- Model interactions between macro drivers, financing conditions, housing supply, and price returns.
- Compare simulated housing outcomes to a renter-investor portfolio baseline.

## Current Snapshot (March 2026)

- Core dataset: `data/processed_data.csv`
- Dataset shape: 675 rows x 97 columns
- Date coverage: 1969-01-01 to 2025-03-01
- Main simulator class: `scripts/market_simulator.py`
- Primary target variable: `Log_Return_MoM`
- Forecast horizon used in production scripts: 300 months (25 years)

## Modeling Architecture

The simulator in `scripts/market_simulator.py` implements a 4-tier pipeline.

1. Tier 1 (macro foundation): ARIMA-style models for independent macro variables.
2. Tier 2 (financial and labor): SARIMAX models conditioned on Tier 1 outputs.
3. Tier 3 (housing supply): SARIMAX models conditioned on Tier 1 and Tier 2 outputs.
4. Tier 4 (price returns): XGBoost predicts `Log_Return_MoM` recursively with engineered features.

Price levels are then reconstructed by cumulatively summing predicted log returns from a starting anchor price.

### Feature Engineering Strategy

The project relies heavily on engineered temporal features, including:

- Lag features (`_lag_N`)
- Delta features (`_delta_N`)
- Rolling average features (`_RA_N`)
- Monthly cyclic features (`month_sin`, `month_cos`)

Feature updates during forecasting are handled by `_update_lags_and_deltas(...)` in `scripts/market_simulator.py`.

### How Features Are Used By Each Model

- Tier 1 (`ARIMA`): forecasts core macro series such as GDP growth, inflation, and population growth.
- Tier 2 (`SARIMAX`): forecasts rates and labor variables using Tier-1 outputs as exogenous drivers.
- Tier 3 (`SARIMAX`): forecasts housing supply and migration variables using combined Tier-1 and Tier-2 outputs.
- Tier 4 (`XGBoost`): predicts monthly housing return (`Log_Return_MoM`) using broad processed features (base + engineered), then returns are accumulated into a price path.

Current feature-selection behavior in code:

- `XGBoost` uses 80 columns in total under current exclusion rules.
- 24 of those are non-lag, non-delta, non-RA base features.
- `Affordability_Ratio_MoM` and `Affordability_Deviation` are maintained for simulation state updates but are excluded from model training in current `exclude_cols` logic.

### Examples of Non-Lag / Non-Delta / Non-RA Features

The examples below are direct columns from `data/processed_data.csv` and do not include `_lag_`, `_delta_`, or `_RA_` suffixes.

| Feature | Interpretation | Typical Model Use |
|---|---|---|
| `GDP_Growth_YoY` | Annual real economic growth rate | Tier 1 (forecast), then Tier 2/3 exogenous input |
| `most_recent_quarterly_gdp_%_change_extended` | Higher-frequency GDP momentum proxy | Tier 1 + XGBoost signal |
| `Inflation_Rate_YoY` | Annual inflation pressure | Tier 1, then rates/supply propagation |
| `Inflation_Rate_MoM` | Monthly inflation movement | Tier 1 + short-horizon pressure signal |
| `National_Pop_Growth_YoY` | National demographic growth trend | Tier 1 + downstream demand context |
| `Municipal_Pop_Growth_YoY` | Toronto-area demographic growth trend | Tier 1 + local demand context |
| `3_month_t_bill` | Short-end policy-sensitive interest rate | Tier 2 financial conditions |
| `5y_bond` | Medium-term bond yield | Tier 2 financing curve context |
| `yield_curve_slope` | Spread between rates on different tenors | Tier 2 macro-cycle indicator |
| `variable_mortgage_rate` | Borrowing cost for variable-rate mortgages | Tier 2 housing affordability pressure |
| `5_year_fixed_mortgage_qualifying_rate` | Stress-test style mortgage qualifier | Tier 2 financing constraint |
| `labour_force_participation_rate` | Labor market participation | Tier 2 labor strength signal |
| `total_employment_rate` | Employment condition proxy | Tier 2 labor/income support |
| `Income_Growth_YoY` | Annual income growth | Tier 2 + affordability dynamics |
| `Labour_Force_Growth_YoY` | Annual labor-force growth | Tier 2 growth capacity proxy |
| `national_debt_to_gdp` | National debt burden relative to output | Tier 2 macro-financial regime |
| `provincial_debt_to_gdp` | Provincial debt burden relative to output | Tier 2 regional macro constraint |
| `Migration_Rate` | Net migration normalized by population | Tier 3 demand/supply pressure bridge |
| `NPR_Rate` | Non-permanent resident rate | Tier 3 demand pressure and volatility |
| `housing_starts_per_cap` | New starts normalized by population | Tier 3 supply expansion signal |
| `under_construction_per_cap` | Pipeline of homes under construction | Tier 3 forward supply pressure |
| `completions_per_cap` | Completed housing stock flow | Tier 3 realized supply signal |
| `month_sin` | Cyclic month-of-year encoding | XGBoost seasonality representation |
| `month_cos` | Complementary cyclic month encoding | XGBoost seasonality representation |

Notes:

- `Log_Return_MoM` is the forecast target, not an input feature in `XGBoost` training.
- Base features capture level and macro-regime information, while lag/delta/RA features capture temporal memory and acceleration effects.

### Stochastic and Scenario Controls

The simulator includes configurable behavior for:

- Tier-1 noise scaling
- Sentiment drift and mean reversion

Operational note: current defaults in code can differ from historical notes in markdown docs. Always treat `scripts/market_simulator.py` as the source of truth for active parameters.

## Repository Layout

```text
.
|- data/
|  |- processed_data.csv
|  \- unprocessed/
|- scripts/
|  |- market_simulator.py
|  |- run_production_forecast.py
|  |- validate_walk_forward.py
|  |- all_model_comp.py
|  |- rent_model.py
|  |- tuning/
|  |  |- preprocessing.py
|  |  |- tune_xgboost.py
|  |  \- correlation_analysis.py
|  \- testbenches/
|     |- audit_updated.py
|     |- validate_longshot.py
|     |- validate_rolling.py
|     \- audit_model.py
|- results/
|  |- v1.0/
|  |- v1.1/
|  |- v1.2-FINAL/
|  \- old/
|- test_*.py
|- diagnose_*.py
\- *.md (enhancement and fix documentation)
```

## Environment Setup

Use Python 3.10+ (the dev container currently runs Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pmdarima statsmodels seaborn
```

Why extra installs are needed:

- `requirements.txt` includes core ML packages (numpy/pandas/scikit-learn/xgboost/matplotlib).
- Core scripts also import `pmdarima`, `statsmodels`, and `seaborn`.

## Data Pipeline

### 1) Build processed features from raw input

Script: `scripts/tuning/preprocessing.py`

```bash
python scripts/tuning/preprocessing.py
```

Preprocessing input note:

- The script reads `data/dataset-MVP-Mar2025.csv`.
- In this repository snapshot, raw files are under `data/unprocessed/`.
- If needed, copy the expected file before running preprocessing:

```bash
cp data/unprocessed/dataset-MVP-Mar2025.csv data/dataset-MVP-Mar2025.csv
```

This script:

- normalizes column names
- interpolates selected raw series
- creates growth, migration, affordability, and cyclic features
- creates lag/delta/rolling-window features
- writes `data/processed_data.csv`

### 2) Optional correlation exploration

Script: `scripts/tuning/correlation_analysis.py`

```bash
python scripts/tuning/correlation_analysis.py
```

Output: `correlation_heatmap.png`.

## Main Workflows

### A) Production Monte Carlo forecast

Script: `scripts/run_production_forecast.py`

```bash
python scripts/run_production_forecast.py
```

Typical outputs (when run in root `results/` path):

- `results/final_simulations.csv`
- `results/final_summary_stats.csv`
- `results/final_simulations_plot.png`

Note: existing committed outputs are currently organized in versioned folders such as `results/v1.0/`, `results/v1.1/`, and `results/v1.2-FINAL/`.

### B) Walk-forward validation

Script: `scripts/validate_walk_forward.py`

```bash
python scripts/validate_walk_forward.py
```

This performs rolling-origin validation across multiple historical folds and computes metrics like MAPE, RMSE, directional accuracy, CAGR error, and distribution coverage.

### C) Single-run full forecast audit

Script: `scripts/testbenches/audit_updated.py`

```bash
python scripts/testbenches/audit_updated.py
```

This produces a long-horizon forecast table with derived features and visuals, useful for inspecting lag/delta/rolling-feature persistence in the forecast period.

### D) Buy-vs-rent comparison visualization

Script: `scripts/all_model_comp.py`

```bash
python scripts/all_model_comp.py
```

Combines:

- simulation percentiles (bear/base/bull)
- historical reconstructed price path
- deterministic average-log-return path
- renter portfolio path from `scripts/rent_model.py`

Output: `results/all_model_comp_plot.png`.

### E) Hyperparameter tuning

Script: `scripts/tuning/tune_xgboost.py`

```bash
python scripts/tuning/tune_xgboost.py
```

Writes tuning outputs such as:

- `results/tuning_results.csv`
- `results/tuning_heatmap.png`

## Tests and Diagnostics

This repository contains many standalone diagnostic scripts rather than a single unified pytest suite.

### Commonly used checks

- `test_diagnostics.py`: variable availability and fit diagnostics
- `test_lag_fix.py`: verifies lag update behavior month-to-month
- `test_lag_delta_update.py`: verifies derived feature persistence
- `test_monthly_seasonality.py`: validates monthly cyclic feature generation
- `test_forecast_features.py`: inspects feature population in forecast CSVs
- `quick_diagnostic.py`: checks for frozen lag-column bug patterns

### Deep diagnostics

- `diagnose_exponential_bias.py`: inspects bias sources (feature importance, drift, return bias)
- `diagnose_gravity_override.py`: traces affordability/gravity logic behavior

### Important caveat on legacy scripts

Some test files reflect older interfaces (for example older constructor signatures, old column names like `Date`, or assumptions that `forecast_price(...)` returns a plain DataFrame). If a test fails immediately with an API mismatch, treat it as legacy and cross-check against current behavior in `scripts/market_simulator.py`.

## Output Artifacts

### Versioned result snapshots

Current repository snapshots include:

- `results/v1.0/`
- `results/v1.1/`
- `results/v1.2-FINAL/`

These contain combinations of:

- `final_simulations.csv`
- `final_summary_stats.csv`
- `forecast_25_year_march2050.csv`
- forecast plots and comparison plots

Historical archives are stored under `results/old/`.

## Documentation Map

Project notes and change logs:

- `ENHANCEMENT_NOTES.md`
- `OPTIMISM_TUNING.md`
- `FIX_SUMMARY.md`
- `LAG_FIX_DETAILED.md`

These docs are useful for historical context and rationale, especially around:

- lag/delta/rolling feature persistence fixes
- garbage-value lag overwrite fix
- sentiment optimism tuning
- tail-risk scenario design

## Known Operational Notes

- Several scripts use hardcoded absolute paths under `/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/...`.
- If you run this project in a different filesystem location, update those constants first.
- `scripts/testbenches/audit_simple.py` references `scripts.visualization_helper`, but the source file is not currently present in the tree.
- `requirements.txt` does not include all imported libraries used by every script.

## Suggested Run Order (Fresh Clone)

1. Install dependencies.
2. Rebuild processed data with `scripts/tuning/preprocessing.py` if needed.
3. Run `scripts/run_production_forecast.py` for baseline output artifacts.
4. Run `scripts/validate_walk_forward.py` for historical robustness checks.
5. Run targeted diagnostics (`test_lag_fix.py`, `test_lag_delta_update.py`, `diagnose_exponential_bias.py`) when debugging behavior.
6. Run `scripts/all_model_comp.py` to produce the buy-vs-rent comparison chart.

## Disclaimer

This project is an educational/research forecasting framework, not financial advice. Long-horizon housing forecasts are sensitive to assumptions, data revisions, and structural regime changes.
