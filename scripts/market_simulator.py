import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try imports, but allow script to define class even if libraries are missing (for testing structure)
try:
    import pmdarima as pm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from xgboost import XGBRegressor
except ImportError:
    pm = None
    SARIMAX = None
    XGBRegressor = None

class MarketSimulator:
    """
    4-Tier Monte Carlo Housing Market Simulator.
    Fixes applied: Infinity handling, Target Leakage prevention, Explicit Level Reconstruction.
    """

    def __init__(self, df: pd.DataFrame, seed: int = 42, start_market_price: float = 1090326.0, xgb_n_estimators: int = 200, xgb_learning_rate: float = 0.05):
        self.df = df.copy()
        
        # --- 1. DATA PREP: Force Date Index & Frequency ---
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.set_index('date', inplace=True)
        
        # Force Month Start (MS) frequency for ARIMA/SARIMAX
        if not self.df.index.freq:
            self.df = self.df.asfreq('MS')
            self.df = self.df.ffill()  # Fill gaps created by resampling

        self.seed = seed
        np.random.seed(self.seed)
        
        # --- XGBoost Hyperparameters ---
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_learning_rate = xgb_learning_rate

        # --- 2. VARIABLE DEFINITIONS ---
        # --- TIER 1: The Independent Macro Foundation (Modeled via ARIMA) ---
        # These variables are driven by federal policy or global macro forces.
        self.tier1_vars = [
            'GDP_Growth_YoY', 
            'most_recent_quarterly_gdp_%_change_extended',
            'Inflation_Rate_YoY',
            'Inflation_Rate_MoM',
            'National_Pop_Growth_YoY', 
            'Municipal_Pop_Growth_YoY'
        ]

        # --- TIER 2: The Financial & Labor Engine (Modeled via SARIMAX, Exog = Tier 1) ---
        # The bond market and labor market react to GDP and Inflation.
        self.tier2_vars = [
            '3_month_t_bill', 
            '5y_bond', 
            'yield_curve_slope',
            'variable_mortgage_rate', 
            '5_year_fixed_mortgage_qualifying_rate',
            'labour_force_participation_rate', 
            'total_employment_rate',
            'Income_Growth_YoY', 
            'national_debt_to_gdp',
            'provincial_debt_to_gdp',
            'Labour_Force_Growth_YoY',
        ]

        # --- TIER 3: The Physical Housing Supply (Modeled via SARIMAX, Exog = Tier 1 + 2) ---
        # Builders decide to pour concrete based on Population (Tier 1) and Borrowing Costs (Tier 2).
        self.tier3_vars = [
            'housing_starts_per_cap', 
            'under_construction_per_cap',
            'completions_per_cap',
            'Migration_Rate', 
            'NPR_Rate'
        ]

        # Target variable is monthly log return (changed in processed_data)
        self.price_col = 'Log_Return_MoM'

        # Starting market price: accept explicit argument, default to March 2025 price
        self.start_market_price = float(start_market_price)

        # Models containers
        self.arima_models = {}
        self.sarimax_models = {}
        self.xgb_model = None
        self.feature_columns = None
        
        # --- 3. EXTREME EVENT CONFIGURATION ---
        # Noise reduction for Tier 1 variables (macro indicators)
        self.tier1_noise_scale = 4.0  # 2% residual noise for realistic variation
        
        # Refined Black Swan: Lower probability, moderate impact, auto-recovery
        self.black_swan_prob = 0  # Disabled - use only model dynamics
        
        # Doomsday Scenario: Catastrophic tail risk (0.0417% per month = 1 in 2,400 = ~1 every 200 years)
        self.doomsday_prob = 0 #0.0004167  # Increased magnitude (was 0.00001)
        self.doomsday_crash_magnitude = 0.18  # 15-20% crash over 24 months
        self.doomsday_duration_months = 24  # 2-year unfolding period
        
        # --- SENTIMENT & BIAS CONFIGURATION (Market Optimism Tuning) ---
        # Baseline + 20% optimism increase from neutral
        # Restores baseline, then adds subtle upward bias (0.1% monthly drift)
        # The "Steady Real Estate" Configuration
        self.sentiment_shock_mean = 0.000    # +0.1% monthly bias (~1.2% annualized upward drift)
        self.sentiment_shock_std = 0.005     # 0.020 for realistic numbers
        self.sentiment_mean_reversion = 0.75 # Shocks fade out quickly (prevents 10-year death spirals)

    def fit(self, train_df: pd.DataFrame = None):
        """Fit all tiers: ARIMA (Tier 1), SARIMAX (Tier 2/3), XGBoost (Tier 4)."""
        if train_df is None:
            train_df = self.df.copy()

        # Variables that have strong seasonal patterns and need seasonal ARIMA
        seasonal_variables = {
           # 'Migration_Rate', 'NPR_Rate',  # Tier 1
           # 'labour_force_participation_rate', 'total_employment_rate',  # Tier 2
           # 'housing_starts_per_cap', 'under_construction_per_cap', 'completions_per_cap'  # Tier 3
        }

        # --- Tier 1: ARIMA for independent macro variables ---
        print("Fitting Tier 1 (ARIMA)...")
        tier1_success = 0
        tier1_failed = []
        for v in self.tier1_vars:
            if v not in train_df.columns: 
                tier1_failed.append(f"{v} (not in data)")
                continue
            series = train_df[v].dropna()
            # Safety check for data length
            if len(series) < 24 or pm is None:
                self.arima_models[v] = None
                tier1_failed.append(f"{v} (insufficient data: {len(series)} rows)")
                continue
            
            try:
                # Use seasonal ARIMA only for variables with strong seasonality
                use_seasonal = v in seasonal_variables
                if False: #use_seasonal:
                    model = pm.auto_arima(series, seasonal=True, m=12,
                                          error_action='ignore', suppress_warnings=True, 
                                          stepwise=True, max_p=2, max_q=2, max_P=1, max_Q=1,
                                          trace=False, n_jobs=-1)
                else:
                    model = pm.auto_arima(series, seasonal=False,
                                          error_action='ignore', suppress_warnings=True, 
                                          stepwise=True, max_p=2, max_q=2, max_d=1,
                                          trace=False, n_jobs=-1)
                self.arima_models[v] = model
                tier1_success += 1
                seasonal_str = "(seasonal)" if use_seasonal else ""
                print(f"    ✓ {v} {seasonal_str}")
            except Exception as e:
                self.arima_models[v] = None
                tier1_failed.append(f"{v} ({str(e)[:50]})")
        
        print(f"  ✓ Tier 1: {tier1_success}/{len(self.tier1_vars)} models fitted successfully")
        if tier1_failed:
            print(f"  ✗ Failed: {', '.join(tier1_failed)}")

        # --- Tier 2 & 3: SARIMAX with exogenous variables ---
        print("Fitting Tier 2 & 3 (SARIMAX)...")
        # Prepare exogenous data (Tier 1) - only use variables that exist and were fitted
        available_tier1_vars = [v for v in self.tier1_vars if v in train_df.columns and self.arima_models.get(v) is not None]
        if not available_tier1_vars:
            print("  ⚠️  No Tier 1 variables available for Tier 2/3 SARIMAX models")
            available_tier1_vars = [v for v in self.tier1_vars if v in train_df.columns]

        exog_tier1 = train_df[available_tier1_vars].ffill().fillna(0)
        
        # Fit Tier 2 (Financial & Labor variables with Tier 1 as exog)
        tier2_success = 0
        tier2_failed = []
        for v in self.tier2_vars:
            if v not in train_df.columns:
                tier2_failed.append(f"{v} (not in data)")
                continue
            try:
                endog = train_df[v].dropna()
                exog = exog_tier1.loc[endog.index]
                # Use seasonal SARIMAX for variables with strong seasonality
                use_seasonal = v in seasonal_variables
                if False: #use_seasonal:
                    mod = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False)
                else:
                    mod = SARIMAX(endog, exog=exog, order=(1, 1, 1), enforce_stationarity=False)
                self.sarimax_models[v] = mod.fit(disp=False, maxiter=150)
                tier2_success += 1
            except Exception as e:
                self.sarimax_models[v] = None
                tier2_failed.append(f"{v} ({str(e)[:50]})")
        
        print(f"  ✓ Tier 2: {tier2_success}/{len(self.tier2_vars)} models fitted successfully")
        if tier2_failed:
            print(f"  ✗ Failed: {', '.join(tier2_failed)}")

        # Fit Tier 3 (Housing Supply with Tier 1 + Tier 2 as exog)
        available_tier2_vars = [v for v in self.tier2_vars if v in train_df.columns]
        exog_tier1_2 = pd.concat([train_df[available_tier1_vars], train_df[available_tier2_vars]], axis=1).ffill().fillna(0)
        tier3_success = 0
        tier3_failed = []
        for v in self.tier3_vars:
            if v not in train_df.columns:
                tier3_failed.append(f"{v} (not in data)")
                continue
            try:
                endog = train_df[v].dropna()
                exog = exog_tier1_2.loc[endog.index]
                # Use seasonal SARIMAX for variables with strong seasonality
                use_seasonal = v in seasonal_variables
                if False: #use_seasonal:
                    mod = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False)
                else:
                    mod = SARIMAX(endog, exog=exog, order=(1, 1, 1), enforce_stationarity=False)
                self.sarimax_models[v] = mod.fit(disp=False, maxiter=150)
                tier3_success += 1
            except Exception as e:
                self.sarimax_models[v] = None
                tier3_failed.append(f"{v} ({str(e)[:50]})")
        
        print(f"  ✓ Tier 3: {tier3_success}/{len(self.tier3_vars)} models fitted successfully")
        if tier3_failed:
            print(f"  ✗ Failed: {', '.join(tier3_failed)}")
        
        # Store available variables for simulation
        self.available_tier1_vars = available_tier1_vars
        self.available_tier2_vars = available_tier2_vars

        # --- Tier 4: XGBoost with simplified feature selection ---
        print("Fitting Tier 4 (XGBoost)...")
        
        # Use all columns from processed_data EXCEPT target/date and leakage-prone state trackers
        # Keep Affordability_Ratio_MoM in the dataframe for simulation/state updates,
        # but exclude it from model training features.
        exclude_cols = {self.price_col, 'date', 'Affordability_Ratio_MoM', 'Affordability_Ratio_MoM_delta_12', 'Affordability_Ratio_MoM_lag_1', 'Affordability_Ratio_MoM_lag_3', 'Affordability_Ratio_MoM_RA_12', 'Affordability_Ratio_MoM_RA_24', 'Log_Return_MoM_RA_3', 'Log_Return_MoM_lag_1', 'Log_Return_MoM_lag_3', 'Log_Return_MoM_lag_6', 'Log_Return_MoM_lag_12', 'Affordability_Deviation_lag_1', 'Affordability_Deviation_lag_3', 'Affordability_Deviation_delta_3','Affordability_Deviation'}
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        # Drop rows where target is NaN
        feature_df = train_df.dropna(subset=[self.price_col])
        
        # Extract features and target
        X = feature_df[feature_cols].fillna(0)
        y = feature_df[self.price_col]
        
        # Handle infinities
        X = X.replace([np.inf, -np.inf], 0)
        
        self.feature_columns = feature_cols
        
        print(f"  XGBoost: {len(feature_cols)} features, {len(X)} samples")
        
        if XGBRegressor:
            self.xgb_model = XGBRegressor(n_estimators=self.xgb_n_estimators, learning_rate=self.xgb_learning_rate, #reg_lambda=20.0,
            max_depth=5, subsample=0.7, colsample_bytree=0.7,n_jobs=-1, random_state=self.seed, verbosity=0)
            self.xgb_model.fit(X, y)
        
        print("Training Complete.")

    def simulate_exogenous(self, steps: int = 300):
        """Simulate Tier 1, 2, 3 exogenous variables from fitted models."""
        last_date = self.df.index.max()
        future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

        def _deterministic_fallback(series: pd.Series, steps: int) -> np.ndarray:
            """Fallback path based only on historical values (no random shocks)."""
            hist = series.dropna()
            if hist.empty:
                return np.zeros(steps)

            last_value = float(hist.iloc[-1])
            if len(hist) >= 6:
                recent_deltas = hist.diff().dropna().tail(6)
                drift = float(recent_deltas.mean()) if not recent_deltas.empty else 0.0
            elif len(hist) >= 2:
                drift = float(hist.iloc[-1] - hist.iloc[-2])
            else:
                drift = 0.0

            step_index = np.arange(1, steps + 1, dtype=float)
            return last_value + drift * step_index
        
        # --- Tier 1 Simulation with REDUCED NOISE ---
        tier1_sim = pd.DataFrame(index=future_index)
        for v in self.tier1_vars:
            model = self.arima_models.get(v)
            if model:
                # Predict and add SCALED noise from residuals
                #try:
                fc = model.predict(n_periods=steps)
                resid = model.resid()
                last_real_value = self.df[v].dropna().iloc[-1]
                # Bootstrap residuals, then scale down by self.tier1_noise_scale
                raw_noise = np.random.choice(4*resid, size=steps, replace=True)
                noise = pd.Series(raw_noise).ewm(span=3).mean().values
                tier1_sim[v] = fc + self.tier1_noise_scale * noise  # REDUCED NOISE
                #except Exception:
                    # Deterministic fallback for tier 1 if ARIMA prediction fails
                    #tier1_sim[v] = _deterministic_fallback(self.df[v], steps)
            else:
                # Deterministic fallback for tier 1 if model wasn't fitted
                #tier1_sim[v] = noise #_deterministic_fallback(self.df[v], steps)
                print("NOISE")
        
        # --- DOOMSDAY SCENARIO: Catastrophic tail risk (0.001% per month) ---
        doomsday_triggered = np.random.random() < self.doomsday_prob
        doomsday_start_month = None
        
        if doomsday_triggered and steps > self.doomsday_duration_months:
            # Doomsday occurs at a random month with enough runway
            doomsday_start_month = np.random.randint(0, steps - self.doomsday_duration_months)
            print(f"⚠️  DOOMSDAY SCENARIO TRIGGERED at month {doomsday_start_month}")
            
            # Apply cascading collapse to Tier 1 variables
            for month_offset in range(self.doomsday_duration_months):
                crisis_month_idx = doomsday_start_month + month_offset
                if crisis_month_idx >= steps:
                    break
                
                # Severity decreases over time (front-loaded crash)
                severity_decay = 1.0 - (month_offset / self.doomsday_duration_months) ** 2
                
                # GDP shock: severe decline in first year, partial recovery in second
                if month_offset < 12:
                    gdp_shock = -0.08 * severity_decay  # Up to -8% GDP
                else:
                    gdp_shock = -0.02 * severity_decay  # Stabilizing year 2
                tier1_sim.iloc[crisis_month_idx, tier1_sim.columns.get_loc('GDP_Growth_YoY')] = gdp_shock
                
                # Population/migration shocks (people leave crisis zones)
                if 'National_Pop_Growth_YoY' in tier1_sim.columns:
                    pop_shock = -0.03 * severity_decay
                    tier1_sim.iloc[crisis_month_idx, tier1_sim.columns.get_loc('National_Pop_Growth_YoY')] = pop_shock
                
                # Inflation spike early (supply chain destruction), then normalizes
                if 'Inflation_Rate_YoY' in tier1_sim.columns:
                    if month_offset < 6:
                        inflation_shock = 0.08  # Inflation spike
                    else:
                        inflation_shock = 0.02 * severity_decay  # Deflation risk
                    tier1_sim.iloc[crisis_month_idx, tier1_sim.columns.get_loc('Inflation_Rate_YoY')] = inflation_shock
            

        
        # --- BLACK SWAN EVENTS: Refined - Rare regime shifts (0.1% per month) ---
        # Lower probability than doomsday but quicker resolution
        black_swan_months = [i for i in range(steps) if np.random.random() < self.black_swan_prob]
        for month_idx in black_swan_months:
            # Skip if this month is in doomsday crisis (avoid double-shocking)
            if doomsday_start_month is not None:
                if doomsday_start_month <= month_idx < doomsday_start_month + self.doomsday_duration_months:
                    continue
            
            # Moderate GDP shock with faster recovery
            tier1_sim.iloc[month_idx, tier1_sim.columns.get_loc('GDP_Growth_YoY')] = -0.03
            

        def _get_idiosyncratic_volatility(model, fallback_std: float) -> float:
            """Estimate forecast volatility from model MSE, with robust fallback."""
            try:
                mse_attr = getattr(model, 'mse', None)
                if mse_attr is not None:
                    mse_value = float(np.nanmean(np.asarray(mse_attr, dtype=float)))
                    if np.isfinite(mse_value) and mse_value > 0:
                        return float(np.sqrt(mse_value))
            except Exception:
                pass
            return float(fallback_std)



        # --- Tier 2 Simulation (SARIMAX forecast) ---
        tier2_sim = pd.DataFrame(index=future_index)
        # Use only variables that were available during fitting
        available_tier1 = getattr(self, 'available_tier1_vars', [v for v in self.tier1_vars if v in tier1_sim.columns])
        exog_tier1 = tier1_sim[available_tier1].fillna(0)
        
        tier2_forecast_failures = []
        for v in self.tier2_vars:
            model = self.sarimax_models.get(v)
            if model:
                try:
                    pred = model.get_forecast(steps=steps, exog=exog_tier1).predicted_mean
                    volatility = _get_idiosyncratic_volatility(model, fallback_std=0.5)
                    idiosyncratic_shocks = np.random.normal(0, volatility * 1.5, size=steps)
                    tier2_sim[v] = pred.values + idiosyncratic_shocks
                except Exception as e:
                    tier2_forecast_failures.append(f"{v}: {str(e)[:40]}")
                    # Deterministic fallback based only on historical trajectory
                    tier2_sim[v] = _deterministic_fallback(self.df[v], steps)
            else:
                # Model wasn't fitted - deterministic fallback from history
                tier2_forecast_failures.append(f"{v}: model not fitted")
                tier2_sim[v] = _deterministic_fallback(self.df[v], steps)
        
        if tier2_forecast_failures:
            print(f"  ⚠️  Tier 2 forecast issues: {', '.join(tier2_forecast_failures)}")

        # --- Tier 3 Simulation (SARIMAX forecast) ---
        tier3_sim = pd.DataFrame(index=future_index)
        # Use only variables that were available during fitting
        available_tier2 = getattr(self, 'available_tier2_vars', [v for v in self.tier2_vars if v in tier2_sim.columns])
        exog_combined = pd.concat([tier1_sim[available_tier1], tier2_sim[available_tier2]], axis=1).fillna(0)
        # Ensure only columns used during fit are passed
        valid_exog_cols = [c for c in exog_combined.columns if c in available_tier1 + available_tier2]
        tier3_forecast_failures = []
        for v in self.tier3_vars:
            model = self.sarimax_models.get(v)
            if model:
                try:
                    pred = model.get_forecast(steps=steps, exog=exog_combined[valid_exog_cols]).predicted_mean
                    volatility = _get_idiosyncratic_volatility(model, fallback_std=0.3)
                    idiosyncratic_shocks = np.random.normal(0, volatility * 1.2, size=steps)
                    tier3_sim[v] = pred.values + idiosyncratic_shocks
                except Exception as e:
                    tier3_forecast_failures.append(f"{v}: {str(e)[:40]}")
                    # Deterministic fallback based only on historical trajectory
                    tier3_sim[v] = _deterministic_fallback(self.df[v], steps)
            else:
                # Model wasn't fitted - deterministic fallback from history
                tier3_forecast_failures.append(f"{v}: model not fitted")
                tier3_sim[v] = _deterministic_fallback(self.df[v], steps)
        
        if tier3_forecast_failures:
            print(f"  ⚠️  Tier 3 forecast issues: {', '.join(tier3_forecast_failures)}")

        return pd.concat([tier1_sim, tier2_sim, tier3_sim], axis=1)

    def forecast_price(self, iterations: int = 100, steps: int = 300):
        """Recursive XGBoost Loop with Market Sentiment & Bubbles."""
        if self.xgb_model is None:
            raise RuntimeError("Model not fitted.")

        all_paths = pd.DataFrame()
        base_hist = self.df.copy()

        print(f"Running {iterations} Monte Carlo iterations (with sentiment/bubbles)...")
        
        for i in range(iterations):
            if i % 10 == 0:
                print(f"  Iteration {i}/{iterations}")
            
            # 1. Generate new world
            sim_world = self.simulate_exogenous(steps=steps)
            
            # 2. Prepare History (clone base)
            current_hist = base_hist.copy()
            
            prices = []
            
            # --- SENTIMENT ACCUMULATOR: Initialize for this iteration ---
            sentiment_score = 0.0
            
            # 3. Recursive Loop: now the model predicts monthly log RETURNS (Log_Return_MoM)
            for t in range(steps):
                current_date = sim_world.index[t]

                # Get the simulated row (Tiers 1-3)
                sim_row = sim_world.iloc[[t]]

                # Ensure sim_row has all columns from current_hist
                for col in current_hist.columns:
                    if col not in sim_row.columns:
                        sim_row[col] = np.nan

                # Append to history (Target return is NaN for now)
                current_hist = pd.concat([current_hist, sim_row], axis=0)
                current_hist = current_hist.ffill()

                # Update features (Calculates Lags based on t-1, which has data)
                # Use wider lookback window to ensure enough history for all lags
                start_idx = max(0, len(current_hist) - 50)
                tail = current_hist.iloc[start_idx:].copy()
                tail = self._update_lags_and_deltas(tail)

                # Write ONLY the current (last) row's calculated features back to current_hist
                # This prevents overwriting previously calculated lag/delta/RA values
                current_row_features = tail.iloc[-1]
                for col in tail.columns:
                    if any(tag in col for tag in ['_lag_', '_delta_', '_RA_', 'month_']):
                        # Ensure column exists in current_hist
                        if col not in current_hist.columns:
                            current_hist[col] = np.nan
                        current_hist.at[current_date, col] = current_row_features[col]

                # Extract the row to predict (the very last one)
                try:
                    X_row = current_hist.iloc[[-1]][self.feature_columns]
                except KeyError:
                    # Some feature columns may not exist, handle gracefully
                    X_row = current_hist.iloc[[-1]].copy()
                    for col in self.feature_columns:
                        if col not in X_row.columns:
                            X_row[col] = 0

                # --- CRITICAL FIX: Handle NaNs and Infs ---
                X_row = X_row.fillna(0).replace([np.inf, -np.inf], 0)

                # --- TIER 4: Predict monthly log RETURN (from XGBoost) ---
                rational_log_return = float(self.xgb_model.predict(X_row)[0])

                # --- MARKET SENTIMENT & BUBBLES (applied to returns) with CONFIGURABLE OPTIMISM ---
                # Generate sentiment shock from configurable distribution
                # - sentiment_shock_mean controls optimism bias (positive = bullish)
                # - sentiment_shock_std controls volatility
                monthly_shock = np.random.normal(self.sentiment_shock_mean, self.sentiment_shock_std)
                sentiment_score = (sentiment_score * self.sentiment_mean_reversion) + monthly_shock

                # Preliminary predicted log return (XGBoost + sentiment)
                pred_log_return = rational_log_return + sentiment_score

                # Maintain cumulative log price for this iteration
                if t == 0:
                    # Initialize from last known price in history, or use start_market_price
                    # We compute log price on-the-fly during simulation
                    current_log_price = float(np.log(self.start_market_price))

                # Update cumulative log price
                current_log_price = current_log_price + pred_log_return

                # Write returns and log price back to history so next step's features can use lags
                current_hist.at[current_date, self.price_col] = pred_log_return
                # Store log price internally for lags (if we ever need it in features)
                if '_Log_Price_Internal' not in current_hist.columns:
                    current_hist['_Log_Price_Internal'] = np.nan
                current_hist.at[current_date, '_Log_Price_Internal'] = current_log_price

                # === CALCULATE AFFORDABILITY CHANGE USING CURRENT PREDICTION ===
                # Now that we have pred_log_return, calculate how this month's price movement
                # affects affordability relative to income growth
                if t > 0:  # Skip first iteration (no previous deviation to reference)
                    try:
                        # Get current month's income growth
                        curr_income_yoy = float(current_hist.at[current_date, 'Income_Growth_YoY'])
                        # Get previous month's affordability deviation
                        prev_deviation = float(current_hist.iloc[-2]['Affordability_Deviation'])
                        
                        # Convert to growth factors
                        # Use PREDICTED price return (not last month's actual)
                        price_growth_factor = np.exp(pred_log_return)
                        # Income growth factor (monthly): (1 + YoY growth)^(1/12)
                        monthly_income_factor = (1.0 + curr_income_yoy) ** (1.0 / 12.0)
                        
                        # Calculate the MoM Affordability Change
                        affordability_change_mom = (price_growth_factor / monthly_income_factor) - 1.0
                        new_deviation = ((1.0 + prev_deviation) * (1.0 + affordability_change_mom)) - 1.0
                        
                        # Write to current month
                        current_hist.at[current_date, 'Affordability_Ratio_MoM'] = affordability_change_mom
                        current_hist.at[current_date, 'Affordability_Deviation'] = new_deviation
                    except (KeyError, TypeError, ValueError, IndexError):
                        # If calculation fails, leave as NaN
                        pass

                prices.append(float(np.exp(current_log_price)))
            
            all_paths[f'iter_{i}'] = pd.Series(prices, index=sim_world.index)

        # Return both price paths and the full feature history from the last iteration
        # The history contains Log_Return_MoM, Affordability_Change_MoM, lags, deltas, RAs
        return {'price_paths': all_paths, 'full_history': current_hist}

    def get_extended_forecast(self, forecast_result) -> pd.DataFrame:
        """Combine forecast results with price path percentiles.
        
        Args:
            forecast_result: Dictionary with 'price_paths' and 'full_history' from forecast_price()
                             OR legacy support for price_paths DataFrame
        """
        # Handle both new dict format and legacy DataFrame format
        if isinstance(forecast_result, dict):
            price_paths = forecast_result['price_paths']
            full_history = forecast_result['full_history']
        else:
            # Legacy format: just price_paths DataFrame
            price_paths = forecast_result
            full_history = None
        
        # If we have the full history with engineered features, use it directly
        if full_history is not None:
            # Return only the forecast portion (after the training data)
            forecast_only = full_history[len(self.df):]
            
            # Add price percentiles
            extended = forecast_only.copy()
            extended['Price_mean'] = np.exp(price_paths.mean(axis=1))
            extended['Price_median'] = np.exp(price_paths.median(axis=1))
            extended['Price_5pct'] = np.exp(price_paths.quantile(0.05, axis=1))
            extended['Price_95pct'] = np.exp(price_paths.quantile(0.95, axis=1))
            return extended
        
        # Fallback: Re-simulate exogenous if full history not available
        sim = self.simulate_exogenous(steps=len(price_paths))
        
        # Calculate price percentiles
        extended = sim.copy()
        extended['Price_mean'] = np.exp(price_paths.mean(axis=1))
        extended['Price_median'] = np.exp(price_paths.median(axis=1))
        extended['Price_5pct'] = np.exp(price_paths.quantile(0.05, axis=1))
        extended['Price_95pct'] = np.exp(price_paths.quantile(0.95, axis=1))
        
        return extended

    def _update_lags_and_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute lags, deltas, and rolling averages for derived features.
        
        For each derived feature column name:
        - Identifies the base column, function type (_lag_, _delta_, or _RA_), and period
        - Applies the corresponding pandas operation:
          * _lag_N: shift(N) - shifts data back N periods
          * _delta_N: diff(N) - calculates absolute difference from N periods ago
          * _RA_N: rolling(window=N).mean() - rolling average over N periods
        """
        out = df.copy()

        # Complete list of derived feature columns to calculate
        derived_features = [
            # Delta features
            '3_month_t_bill_delta_1', '3_month_t_bill_delta_3', '3_month_t_bill_delta_6', '3_month_t_bill_delta_12',
            '5y_bond_delta_1', '5y_bond_delta_3', '5y_bond_delta_6', '5y_bond_delta_12',
            'yield_curve_slope_delta_1', 'yield_curve_slope_delta_3', 'yield_curve_slope_delta_6', 'yield_curve_slope_delta_12',
            'variable_mortgage_rate_delta_1', 'variable_mortgage_rate_delta_3', 'variable_mortgage_rate_delta_6', 'variable_mortgage_rate_delta_12',
            '5_year_fixed_mortgage_qualifying_rate_delta_1', '5_year_fixed_mortgage_qualifying_rate_delta_3', '5_year_fixed_mortgage_qualifying_rate_delta_6', '5_year_fixed_mortgage_qualifying_rate_delta_12',
            'Migration_Rate_delta_6', 'Migration_Rate_delta_12', 'Affordability_Deviation_delta_3', 'Affordability_Ratio_MoM_delta_3', 'Affordability_Ratio_MoM_delta_12',
            # Lag features
            'GDP_Growth_YoY_lag_6',
            'National_Pop_Growth_YoY_lag_6',
            'Municipal_Pop_Growth_YoY_lag_6',
            'Inflation_Rate_YoY_lag_6',
            'Labour_Force_Growth_YoY_lag_6',
            'Income_Growth_YoY_lag_6',
            'Migration_Rate_lag_6', 'Migration_Rate_lag_12',
            'NPR_Rate_lag_6', 'NPR_Rate_lag_12',
            'Log_Return_MoM_lag_1', 'Log_Return_MoM_lag_3', 'Log_Return_MoM_lag_6', 'Log_Return_MoM_lag_12',
           # 'sales_volume_MoM_lag_1', 'sales_volume_MoM_lag_3', 'sales_volume_MoM_lag_12',
           # 'sales_volume_YoY_lag_1', 'sales_volume_YoY_lag_3', 'sales_volume_YoY_lag_12',
            'housing_starts_per_cap_lag_12', 'housing_starts_per_cap_lag_24',
            'under_construction_per_cap_lag_12', 'under_construction_per_cap_lag_24',
            'completions_per_cap_lag_12', 'completions_per_cap_lag_24',
            'Affordability_Ratio_MoM_lag_1', 'Affordability_Ratio_MoM_lag_3',
            'Affordability_Deviation_lag_1', 'Affordability_Deviation_lag_3',
            # Rolling Average features
            'GDP_Growth_YoY_RA_12',
            'National_Pop_Growth_YoY_RA_12',
            'Municipal_Pop_Growth_YoY_RA_12',
            'Affordability_Ratio_MoM_RA_12', 'Affordability_Ratio_MoM_RA_24',
            'Inflation_Rate_YoY_RA_12',
            'Labour_Force_Growth_YoY_RA_12',
            'Income_Growth_YoY_RA_12',
            'Migration_Rate_RA_12',
            'NPR_Rate_RA_12',
            'housing_starts_per_cap_RA_6', 'housing_starts_per_cap_RA_12',
            'under_construction_per_cap_RA_6', 'under_construction_per_cap_RA_12',
            'completions_per_cap_RA_6', 'completions_per_cap_RA_12'
            #'Log_Return_MoM_RA_3', 'Log_Return_MoM_RA_12'
        ]

        if 'Log_Return_MoM' in out.columns:
            out['Log_Return_MoM_RA_12'] = out['Log_Return_MoM'].rolling(window=12).mean()

        # Process each derived feature
        for feature in derived_features:
            if '_lag_' in feature:
                # Parse: base_col_lag_period -> base_col, period
                base_col, period_str = feature.rsplit('_lag_', 1)
                period = int(period_str)
                
                # Calculate lag using shift
                if base_col in out.columns:
                    out[feature] = out[base_col].shift(period)

            elif '_delta_' in feature:
                # Parse: base_col_delta_period -> base_col, period
                base_col, period_str = feature.rsplit('_delta_', 1)
                period = int(period_str)
                
                # Calculate delta using diff
                if base_col in out.columns:
                    out[feature] = out[base_col].diff(period)

            elif '_RA_' in feature:
                # Parse: base_col_RA_period -> base_col, period
                base_col, period_str = feature.rsplit('_RA_', 1)
                period = int(period_str)
                
                # Calculate rolling average using rolling().mean()
                if base_col in out.columns:
                    out[feature] = out[base_col].rolling(window=period).mean()

        # --- UPDATE CYCLICAL SEASONAL FEATURES ---
        # month_sin and month_cos are based on the month number (1-12) from the datetime index
        # They repeat in a perfect 12-month cycle using sin and cos transformations
        # Formula: month_sin = sin(2π * month / 12), month_cos = cos(2π * month / 12)
        
        # Hardcoded lookup table for all 12 months (deterministic pattern)
        month_sin_lookup = {
            1: 0.5, 2: 0.8660254037844387, 3: 1.0, 4: 0.8660254037844387,
            5: 0.5, 6: 1.2246467991473532e-16, 7: -0.5, 8: -0.8660254037844387,
            9: -1.0, 10: -0.8660254037844387, 11: -0.5, 12: -2.4492935982947064e-16
        }
        month_cos_lookup = {
            1: 0.8660254037844387, 2: 0.5, 3: 6.123233995736766e-17, 4: -0.5,
            5: -0.8660254037844387, 6: -1.0, 7: -0.8660254037844387, 8: -0.5,
            9: -1.8369701987210297e-16, 10: 0.5, 11: 0.8660254037844387, 12: 1.0
        }
        
        # Initialize columns if they don't exist
        if 'month_sin' not in out.columns:
            out['month_sin'] = np.nan
        if 'month_cos' not in out.columns:
            out['month_cos'] = np.nan
        
        # Apply the lookup to each row based on its date's month
        try:
            # Try DatetimeIndex first
            if hasattr(out.index, 'month'):
                months = out.index.month
                out['month_sin'] = [month_sin_lookup[m] for m in months]
                out['month_cos'] = [month_cos_lookup[m] for m in months]
            else:
                # Convert to datetime if needed
                datetime_index = pd.to_datetime(out.index)
                months = datetime_index.month
                out['month_sin'] = [month_sin_lookup[m] for m in months]
                out['month_cos'] = [month_cos_lookup[m] for m in months]
        except Exception as e:
            # Fallback: leave as NaN if extraction fails
            pass

        return out


if __name__ == "__main__":
    print("MarketSimulator Class Defined.")
