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

    def __init__(self, df: pd.DataFrame, seed: int = 42, start_market_price: float = 1090326.0):
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

        # --- 2. VARIABLE DEFINITIONS ---
        self.tier1_vars = [
            'GDP_Growth_YoY', 'National_Pop_Growth_YoY', 
            'Provincial_Pop_Growth_YoY', 'Municipal_Pop_Growth_YoY', 
            'Inflation_Rate_YoY'
        ]

        self.tier2_vars = [
            'variable_mortgage_rate', '5_year_fixed_mortgage_rate',
            '5_year_fixed_mortgage_qualifying_rate', '3_month_t_bill',
            '2y_bond', '5y_bond', '10y_bond', 'yield_curve_slope',
            'labour_force_participation_rate', 'total_employment_rate',
            'Income_Growth_YoY', 'national_debt_to_gdp'
        ]

        self.tier3_vars = [
            'housing_starts_per_cap', 'under_construction_per_cap',
            'completions_per_cap', 'Migration_Rate', 'NPR_Rate'
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
        self.tier1_noise_scale = 0.5  # Reduce residual noise by 50% to improve signal-to-noise
        
        # Refined Black Swan: Lower probability, moderate impact, auto-recovery
        self.black_swan_prob = 0.001  # 0.1% per month (was 0.5%)
        
        # Doomsday Scenario: Catastrophic tail risk (0.0417% per month = 1 in 2,400 = ~1 every 200 years)
        self.doomsday_prob = 0 #0.0004167  # Increased magnitude (was 0.00001)
        self.doomsday_crash_magnitude = 0.18  # 15-20% crash over 24 months
        self.doomsday_duration_months = 24  # 2-year unfolding period
        
        # --- SENTIMENT & BIAS CONFIGURATION (Market Optimism Tuning) ---
        # Baseline + 20% optimism increase from neutral
        # Restores baseline, then adds subtle upward bias (0.1% monthly drift)
        self.sentiment_shock_mean = 0.0009  # +0.1% monthly bias (subtle optimism)
        self.sentiment_shock_std = 0.02  # Standard deviation of shocks (volatility of sentiment)
        self.sentiment_mean_reversion = 0.958  # Slightly faster recovery (~0.7% annualized upward drift)

        # Explicit mapping: Growth Rate -> Absolute Level Column
        self.growth_to_level_map = {
            'GDP_Growth_YoY': 'national_gdp_real,_seasonally_adjusted',
            'National_Pop_Growth_YoY': 'national_pop',
            'Provincial_Pop_Growth_YoY': 'provincial_pop',
            'Municipal_Pop_Growth_YoY': 'municipal_pop',
            'Income_Growth_YoY': 'median_income_per_household_in_toronto',
            'Inflation_Rate_YoY': 'cpi___national,_all_products',
        }

    def fit(self, train_df: pd.DataFrame = None):
        """Fit all models (Tier 1 -> Tier 4)."""
        if train_df is None:
            train_df = self.df.copy()

        print("Fitting Tier 1 (ARIMA)...")
        for v in self.tier1_vars:
            if v not in train_df.columns: 
                continue
            series = train_df[v].dropna()
            # Safety check for data length
            if len(series) < 24 or pm is None:
                self.arima_models[v] = None
                continue
            
            try:
                model = pm.auto_arima(series, seasonal=True, m=12, 
                                      error_action='ignore', suppress_warnings=True, 
                                      stepwise=True, max_p=2, max_q=2)
                self.arima_models[v] = model
            except Exception:
                self.arima_models[v] = None

        print("Fitting Tier 2 & 3 (SARIMAX)...")
        # Prepare exogenous data (Tier 1)
        exog_tier1 = train_df[self.tier1_vars].ffill().fillna(0)
        
        # Fit Tier 2
        for v in self.tier2_vars:
            if v not in train_df.columns: 
                continue
            try:
                endog = train_df[v].dropna()
                exog = exog_tier1.loc[endog.index]
                mod = SARIMAX(endog, exog=exog, order=(1, 1, 1), enforce_stationarity=False)
                self.sarimax_models[v] = mod.fit(disp=False)
            except Exception:
                self.sarimax_models[v] = None

        # Fit Tier 3 (Exog = Tier 1 + Tier 2)
        exog_tier1_2 = pd.concat([train_df[self.tier1_vars], train_df[self.tier2_vars]], axis=1).ffill().fillna(0)
        for v in self.tier3_vars:
            if v not in train_df.columns: 
                continue
            try:
                endog = train_df[v].dropna()
                exog = exog_tier1_2.loc[endog.index]
                mod = SARIMAX(endog, exog=exog, order=(1, 1, 1), enforce_stationarity=False)
                self.sarimax_models[v] = mod.fit(disp=False)
            except Exception:
                self.sarimax_models[v] = None

        print("Fitting Tier 4 (XGBoost)...")
        # Feature Engineering on History
        hist = train_df.copy()
        hist = self._update_lags_and_deltas(hist)
        
        # Drop rows where target is NaN
        feature_df = hist.dropna(subset=[self.price_col])

        # Select Features: Lags, Deltas (BUT NOT current target delta)
        candidate_cols = [c for c in feature_df.columns if (
            c.endswith('_lag_1') or c.endswith('_lag_12') or 
            c.endswith('_delta_1m_pct') or c.endswith('_delta_12m_pct')
        )]

        # --- CRITICAL FIX: LEAKAGE PREVENTION ---
        # We cannot use the CURRENT month's price delta to predict the current month's price.
        # We can only use LAGGED price deltas.
        forbidden_cols = [f'{self.price_col}_delta_1m_pct', f'{self.price_col}_delta_12m_pct']
        
        # --- FEATURE REALIZABILITY CHECK ---
        # Exclude columns from preprocessing that were dropped (won't exist in simulation).
        dropped_bases = {
            'market_price_target_average,_detached_single_family_homes',
            'Log_Price',
            'national_pop', 'provincial_pop', 'municipal_pop',
            'national_gdp_real,_seasonally_adjusted',
            'provincial_gdp_real,_seasonally_adjusted',
            'median_income_per_household_in_toronto',
            'cpi___national,_all_products', 'cpi___national,_core',
            'housing_starts_sfh,_monthly', 'under_construction_sfh,_monthly',
            'completions__sfh,_monthly', 'sales_volume',
            'ontario_net_international_migration_monthly',
            'ontario_net_interprovincial_migration_monthly',
            'ontario_net_non_permanent_residents',
        }
        
        realizable_cols = []
        for c in candidate_cols:
            if c in forbidden_cols:
                continue
            # Skip Log_Price features - it's not a simulation variable
            if 'Log_Price' in c:
                continue
            # Extract base name (before _lag_ or _delta_)
            base = None
            for suffix in ['_lag_1', '_lag_12', '_delta_1m_pct', '_delta_12m_pct']:
                if c.endswith(suffix):
                    base = c[:-len(suffix)]
                    break
            if base and base in dropped_bases:
                continue
            realizable_cols.append(c)
        
        self.feature_columns = realizable_cols if realizable_cols else candidate_cols

        if not self.feature_columns:
            raise ValueError("No valid feature columns found for XGBoost.")

        X = feature_df[self.feature_columns].fillna(0)
        y = feature_df[self.price_col]

        # --- CRITICAL FIX: INFINITY HANDLING ---
        # Replace Infs with 0 or NaN, then drop
        X = X.replace([np.inf, -np.inf], 0)
        
        if XGBRegressor:
            self.xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=self.seed, verbosity=0)
            self.xgb_model.fit(X, y)
        
        print("Training Complete.")

    def simulate_exogenous(self, steps: int = 300):
        """Simulate Tier 1, 2, 3 and reconstruct levels with idiosyncratic shocks."""
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

        # Reconstruct Absolute Levels (e.g. GDP Growth -> GDP Level)
        tier1_sim = self._reconstruct_levels(tier1_sim)
        
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
            
            # Rebuild GDP levels after shock sequence
            if 'national_gdp_real,_seasonally_adjusted' in tier1_sim.columns:
                rates = tier1_sim['GDP_Growth_YoY'].values
                last_level = self.df['national_gdp_real,_seasonally_adjusted'].iloc[-1]
                new_levels = [last_level]
                for r in rates:
                    new_levels.append(new_levels[-1] * (1 + r))
                tier1_sim['national_gdp_real,_seasonally_adjusted'] = new_levels[1:]
        
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
            
            # Rebuild the reconstructed GDP level after the shock
            if 'national_gdp_real,_seasonally_adjusted' in tier1_sim.columns:
                rates = tier1_sim['GDP_Growth_YoY'].values
                last_level = self.df['national_gdp_real,_seasonally_adjusted'].iloc[-1]
                new_levels = [last_level]
                for r in rates:
                    new_levels.append(new_levels[-1] * (1 + r))
                tier1_sim['national_gdp_real,_seasonally_adjusted'] = new_levels[1:]

        # --- Tier 2 Simulation with IDIOSYNCRATIC SHOCKS ---
        tier2_sim = pd.DataFrame(index=future_index)
        exog_tier1 = tier1_sim[self.tier1_vars].fillna(0)
        
        for v in self.tier2_vars:
            model = self.sarimax_models.get(v)
            if model:
                try:
                    # Predicting all steps using the simulated exog
                    pred = model.get_forecast(steps=steps, exog=exog_tier1).predicted_mean
                    # --- CRITICAL: INJECT INDEPENDENT IDIOSYNCRATIC SHOCKS ---
                    # Add volatility independent of Tier 1 (policy errors, supply shocks)
                    volatility = np.sqrt(model.mse) if hasattr(model, 'mse') else 0.5
                    idiosyncratic_shocks = np.random.normal(0, volatility * 1.5, size=steps)  # 1.5x for realism
                    tier2_sim[v] = pred.values + idiosyncratic_shocks
                except Exception:
                    tier2_sim[v] = self.df[v].mean()
            else:
                tier2_sim[v] = self.df[v].mean()

        # --- Tier 3 Simulation with IDIOSYNCRATIC SHOCKS ---
        tier3_sim = pd.DataFrame(index=future_index)
        exog_combined = pd.concat([tier1_sim, tier2_sim], axis=1).fillna(0)
        # Ensure only columns used during fit are passed
        valid_exog_cols = [c for c in exog_combined.columns if c in self.tier1_vars + self.tier2_vars]
        
        for v in self.tier3_vars:
            model = self.sarimax_models.get(v)
            if model:
                try:
                    pred = model.get_forecast(steps=steps, exog=exog_combined[valid_exog_cols]).predicted_mean
                    # --- CRITICAL: INJECT INDEPENDENT IDIOSYNCRATIC SHOCKS ---
                    # Add volatility for supply shocks (strikes, shortages, immigration policy)
                    volatility = np.sqrt(model.mse) if hasattr(model, 'mse') else 0.3
                    idiosyncratic_shocks = np.random.normal(0, volatility * 1.2, size=steps)
                    tier3_sim[v] = pred.values + idiosyncratic_shocks
                except Exception:
                    tier3_sim[v] = self.df[v].mean()
            else:
                tier3_sim[v] = self.df[v].mean()

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

                # Append to history (Target return is NaN for now)
                current_hist = pd.concat([current_hist, sim_row])

                # Update features (Calculates Lags based on t-1, which has data)
                tail = current_hist.iloc[-24:].copy()  # Look back enough for 12m lags
                tail = self._update_lags_and_deltas(tail)

                # Extract the row to predict (the very last one)
                X_row = tail.iloc[[-1]][self.feature_columns]

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
                    # Initialize from last known log price if available in history, else use start_market_price
                    if 'Log_Price' in current_hist.columns and not current_hist['Log_Price'].dropna().empty:
                        current_log_price = float(current_hist['Log_Price'].dropna().iloc[-1])
                    elif 'Market_Price' in current_hist.columns and not current_hist['Market_Price'].dropna().empty:
                        current_log_price = float(np.log(current_hist['Market_Price'].dropna().iloc[-1]))
                    else:
                        current_log_price = float(np.log(self.start_market_price))

                # Update cumulative log price
                current_log_price = current_log_price + pred_log_return

                # Write returns and log price back to history so next step's Lag_1 is correct
                current_hist.at[current_date, self.price_col] = pred_log_return
                current_hist.at[current_date, 'Log_Price'] = current_log_price

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

    def _reconstruct_levels(self, sim_df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct absolute levels from growth rates."""
        out = sim_df.copy()
        for rate_col, level_col in self.growth_to_level_map.items():
            if rate_col in out.columns and level_col in self.df.columns:
                # Get last known actual level
                last_level = self.df[level_col].iloc[-1]
                
                # Apply growth rates cumulatively
                # New = Old * (1 + Rate)
                rates = out[rate_col].values
                new_levels = [last_level]
                for r in rates:
                    new_levels.append(new_levels[-1] * (1 + r))
                
                # Assign (skipping the first seed value)
                out[level_col] = new_levels[1:]
        return out

    def _update_lags_and_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lags/deltas dynamically."""
        out = df.copy()
        # We need features for ALL variables (Tiers 1-3 and the return target, but NOT Log_Price which is internal)
        cols_to_lag = [c for c in out.columns if c in self.tier1_vars + self.tier2_vars + self.tier3_vars + [self.price_col] + list(self.growth_to_level_map.values())]
        
        for col in cols_to_lag:
            # Lags
            out[f'{col}_lag_1'] = out[col].shift(1)
            out[f'{col}_lag_12'] = out[col].shift(12)
            
            # Deltas (Handle potential division by zero)
            out[f'{col}_delta_1m_pct'] = out[col].pct_change(1).replace([np.inf, -np.inf], 0)
            out[f'{col}_delta_12m_pct'] = out[col].pct_change(12).replace([np.inf, -np.inf], 0)
            
        return out


if __name__ == "__main__":
    print("MarketSimulator Class Defined.")
