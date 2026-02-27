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
            'Municipal_Pop_Growth_YoY', 
            'Migration_Rate', 
            'NPR_Rate'
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
            'completions_per_cap'
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
        self.tier1_noise_scale = 0.25 
        
        # Refined Black Swan: Lower probability, moderate impact, auto-recovery
        self.black_swan_prob = 0.001  # 0.1% per month (was 0.5%)
        
        # Doomsday Scenario: Catastrophic tail risk (0.0417% per month = 1 in 2,400 = ~1 every 200 years)
        self.doomsday_prob = 0 #0.0004167  # Increased magnitude (was 0.00001)
        self.doomsday_crash_magnitude = 0.18  # 15-20% crash over 24 months
        self.doomsday_duration_months = 24  # 2-year unfolding period
        
        # --- SENTIMENT & BIAS CONFIGURATION (Market Optimism Tuning) ---
        # Baseline + 20% optimism increase from neutral
        # Restores baseline, then adds subtle upward bias (0.1% monthly drift)
        # The "Steady Real Estate" Configuration
        self.sentiment_shock_mean = 0.001    # +0.1% monthly bias (~1.2% annualized upward drift)
        self.sentiment_shock_std = 0.004     # 0.4% monthly volatility (smooths out the erratic bouncing)
        self.sentiment_mean_reversion = 0.25 # Shocks fade out quickly (prevents 10-year death spirals)

    def fit(self, train_df: pd.DataFrame = None):
        """Fit all tiers: ARIMA (Tier 1), SARIMAX (Tier 2/3), XGBoost (Tier 4)."""
        if train_df is None:
            train_df = self.df.copy()

        # Variables that have strong seasonal patterns and need seasonal ARIMA
        seasonal_variables = {
            'Migration_Rate', 'NPR_Rate',  # Tier 1
            'labour_force_participation_rate', 'total_employment_rate',  # Tier 2
            'housing_starts_per_cap', 'under_construction_per_cap', 'completions_per_cap'  # Tier 3
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
                if use_seasonal:
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
                if use_seasonal:
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
                if use_seasonal:
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
        
        # Use all columns from processed_data EXCEPT the target
        exclude_cols = {self.price_col}  # Only exclude Log_Return_MoM (target)
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
            self.xgb_model = XGBRegressor(n_estimators=self.xgb_n_estimators, learning_rate=self.xgb_learning_rate, max_depth=3, 
            subsample=0.7, colsample_bytree=0.7,n_jobs=-1, random_state=self.seed, verbosity=0)
            self.xgb_model.fit(X, y)
        
        print("Training Complete.")

    def simulate_exogenous(self, steps: int = 300):
        """Simulate Tier 1, 2, 3 exogenous variables with idiosyncratic shocks and extreme events."""
        last_date = self.df.index.max()
        future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
        
        # --- Tier 1 Simulation with REDUCED NOISE ---
        tier1_sim = pd.DataFrame(index=future_index)
        for v in self.tier1_vars:
            model = self.arima_models.get(v)
            if model:
                # Predict and add SCALED noise from residuals
                # Scaling by 0.5x preserves patterns while reducing month-to-month noise
                try:
                    fc = model.predict(n_periods=steps)
                    resid = model.resid()
                    # Bootstrap residuals, then scale down by self.tier1_noise_scale
                    noise = np.random.choice(resid, size=steps, replace=True)
                    tier1_sim[v] = fc + self.tier1_noise_scale * noise  # REDUCED NOISE
                except Exception:
                    hist = self.df[v].dropna()
                    tier1_sim[v] = np.random.normal(hist.mean(), hist.std() * self.tier1_noise_scale, size=steps)
            else:
                # Fallback: Random walk with drift
                hist = self.df[v].dropna()
                tier1_sim[v] = np.random.normal(hist.mean(), hist.std() * self.tier1_noise_scale, size=steps)
        
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
            


        # --- Tier 2 Simulation with IDIOSYNCRATIC SHOCKS ---
        tier2_sim = pd.DataFrame(index=future_index)
        # Use only variables that were available during fitting
        available_tier1 = getattr(self, 'available_tier1_vars', [v for v in self.tier1_vars if v in tier1_sim.columns])
        exog_tier1 = tier1_sim[available_tier1].fillna(0)
        
        tier2_forecast_failures = []
        for v in self.tier2_vars:
            model = self.sarimax_models.get(v)
            if model:
                try:
                    # Predicting all steps using the simulated exog
                    pred = model.get_forecast(steps=steps, exog=exog_tier1).predicted_mean
                    # --- CRITICAL: INJECT INDEPENDENT IDIOSYNCRATIC SHOCKS ---
                    # Add volatility independent of Tier 1 (policy errors, supply shocks)
                    volatility = np.sqrt(model.mse) if hasattr(model, 'mse') else 0.5
                    idiosyncratic_shocks = np.random.normal(0, volatility * 2.0, size=steps)  # Increased to 2.0x for more variation
                    tier2_sim[v] = pred.values + idiosyncratic_shocks
                except Exception as e:
                    tier2_forecast_failures.append(f"{v}: {str(e)[:40]}")
                    # Fallback: use random walk based on historical statistics
                    hist = self.df[v].dropna()
                    hist_std = hist.std() if not hist.empty else 0.0
                    tier2_sim[v] = np.random.normal(hist.mean(), hist_std if hist_std > 0 else 0.01, size=steps)
            else:
                # Model wasn't fitted - use historical mean with noise
                tier2_forecast_failures.append(f"{v}: model not fitted")
                hist = self.df[v].dropna()
                hist_std = hist.std() if not hist.empty else 0.0
                tier2_sim[v] = np.random.normal(hist.mean(), hist_std if hist_std > 0 else 0.01, size=steps)
        
        if tier2_forecast_failures:
            print(f"  ⚠️  Tier 2 forecast issues: {', '.join(tier2_forecast_failures)}")

        # --- Tier 3 Simulation with IDIOSYNCRATIC SHOCKS ---
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
                    # --- CRITICAL: INJECT INDEPENDENT IDIOSYNCRATIC SHOCKS ---
                    # Add volatility for supply shocks (strikes, shortages, immigration policy)
                    volatility = np.sqrt(model.mse) if hasattr(model, 'mse') else 0.3
                    idiosyncratic_shocks = np.random.normal(0, volatility * 1.5, size=steps)  # Increased variation
                    tier3_sim[v] = pred.values + idiosyncratic_shocks
                except Exception as e:
                    tier3_forecast_failures.append(f"{v}: {str(e)[:40]}")
                    # Fallback: use random walk based on historical statistics
                    hist = self.df[v].dropna()
                    hist_std = hist.std() if not hist.empty else 0.0
                    tier3_sim[v] = np.random.normal(hist.mean(), hist_std if hist_std > 0 else 0.01, size=steps)
            else:
                # Model wasn't fitted - use historical mean with noise
                tier3_forecast_failures.append(f"{v}: model not fitted")
                hist = self.df[v].dropna()
                hist_std = hist.std() if not hist.empty else 0.0
                tier3_sim[v] = np.random.normal(hist.mean(), hist_std if hist_std > 0 else 0.01, size=steps)
        
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

                # Write all calculated features back to current_hist
                for col in tail.columns:
                    current_hist.loc[tail.index, col] = tail[col]

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

                # Final predicted log return for this month
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

                # --- DYNAMIC AFFORDABILITY RECALCULATION (after prediction) ---
                try:
                    # 1. Get last month's known affordability ratio
                    last_affordability = current_hist['Affordability_Ratio'].dropna().iloc[-1]
                    
                    # 2. Get the current simulated Income Growth (YoY) and de-annualize it to a MoM factor
                    current_income_yoy = current_hist['Income_Growth_YoY'].iloc[-1]
                    monthly_income_factor = (1 + current_income_yoy) ** (1/12)
                    
                    # 3. Convert XGBoost's predicted log return into a simple price growth factor
                    price_growth_factor = np.exp(pred_log_return)
                    
                    # 4. Calculate the new ratio 
                    current_affordability = last_affordability * (price_growth_factor / monthly_income_factor)
                    
                    # 5. Write it back to the history dataframe immediately
                    current_hist.at[current_date, 'Affordability_Ratio'] = current_affordability
                except (KeyError, IndexError):
                    # If affordability can't be calculated, use last known value
                    pass

                prices.append(float(np.exp(current_log_price)))
            
            all_paths[f'iter_{i}'] = pd.Series(prices, index=sim_world.index)

        return all_paths

    def get_extended_forecast(self, price_paths: pd.DataFrame) -> pd.DataFrame:
        """Combine simulated exogenous variables with price path percentiles."""
        # Re-simulate exogenous one more time to get the full picture
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
        Compute lags, deltas, and rolling averages for simulated exogenous variables.
        Matches the feature engineering in processed_data.csv.
        """
        out = df.copy()

        feature_cols = self.feature_columns or []
        lag_periods = {}
        delta_periods = {}
        ra_periods = {}

        for feat in feature_cols:
            if '_lag_' in feat:
                base, period = feat.rsplit('_lag_', 1)
                if period.isdigit():
                    lag_periods.setdefault(base, set()).add(int(period))
            elif '_delta_' in feat:
                base, period = feat.rsplit('_delta_', 1)
                if period.isdigit():
                    delta_periods.setdefault(base, set()).add(int(period))
            elif '_RA_' in feat:
                base, period = feat.rsplit('_RA_', 1)
                if period.isdigit():
                    ra_periods.setdefault(base, set()).add(int(period))

        if not feature_cols:
            base_cols = [c for c in out.columns if not any(tag in c for tag in ['_lag_', '_delta_', '_RA_'])]
            base_cols = [c for c in base_cols if c not in {'date'}]
            lag_periods = {c: {1, 3, 6, 12, 24} for c in base_cols}
            delta_periods = {c: {1, 3, 6, 12} for c in base_cols}
            ra_periods = {c: {6, 12, 24} for c in base_cols}

        for base_col, periods in lag_periods.items():
            if base_col not in out.columns or out[base_col].isna().all():
                continue
            for p in periods:
                out[f'{base_col}_lag_{p}'] = out[base_col].shift(p)

        for base_col, periods in delta_periods.items():
            if base_col not in out.columns or out[base_col].isna().all():
                continue
            for p in periods:
                out[f'{base_col}_delta_{p}'] = out[base_col].pct_change(p).replace([np.inf, -np.inf], 0)

        for base_col, periods in ra_periods.items():
            if base_col not in out.columns or out[base_col].isna().all():
                continue
            for p in periods:
                out[f'{base_col}_RA_{p}'] = out[base_col].rolling(window=p).mean()

        return out


if __name__ == "__main__":
    print("MarketSimulator Class Defined.")
