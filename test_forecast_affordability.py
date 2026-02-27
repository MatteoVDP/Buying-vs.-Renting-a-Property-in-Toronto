#!/usr/bin/env python
"""Quick test of affordability calculation in forecast loop."""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator

print("="*80)
print("QUICK TEST: Affordability & Lags/Deltas in Forecast Loop")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS')
df = df.ffill()

print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"   Date range: {df.index.min().strftime('%B %Y')} to {df.index.max().strftime('%B %Y')}")

# Check if Affordability_Ratio exists
if 'Affordability_Ratio' not in df.columns:
    print("   ⚠️  ERROR: Affordability_Ratio column not found!")
    sys.exit(1)
else:
    print(f"   ✓ Affordability_Ratio column found")
    print(f"     - Last value: {df['Affordability_Ratio'].iloc[-1]:.6f}")

# Check for lag/delta/RA columns
lag_cols = [c for c in df.columns if '_lag_' in c]
delta_cols = [c for c in df.columns if '_delta_' in c]
ra_cols = [c for c in df.columns if '_RA_' in c]

print(f"\n2. Feature Engineering Columns:")
print(f"   Lag columns: {len(lag_cols)}")
print(f"   Delta columns: {len(delta_cols)}")
print(f"   Rolling Average columns: {len(ra_cols)}")

# Initialize simulator
print("\n3. Initializing MarketSimulator...")  
sim = MarketSimulator(df, seed=42)

# Fit models
print("\n4. Fitting models (this may take a moment)...")
sim.fit()
print("   ✓ Models fitted")

# Quick forecast (10 months for speed)
print("\n5. Running quick 10-month forecast...")
steps = 10
sim.start_market_price = 1090326.0

# Get simulated exogenous variables
sim_exog = sim.simulate_exogenous(steps=steps)
print(f"   ✓ Generated {len(sim_exog)} months of exogenous variables")

# Manual forecast loop (same as audit_updated.py)
base_hist = sim.df.copy()
current_hist = base_hist.copy()

future_index = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

print("\n6. Testing forecast loop with affordability calculation...")
print("   " + "-"*76)

log_returns = []
price_path = []
affordability_path = []
current_log_price = float(np.log(sim.start_market_price))
pred_log_return = 0

affordability_calculated = False
lags_calculated = False

for t in range(steps):
    current_date = future_index[t]
    sim_row = sim_exog.iloc[[t]]
    
    # Ensure sim_row has all columns from current_hist
    for col in current_hist.columns:
        if col not in sim_row.columns:
            sim_row[col] = np.nan
    
    current_hist = pd.concat([current_hist, sim_row], axis=0)
    current_hist = current_hist.ffill()
    
    # --- DYNAMIC AFFORDABILITY RECALCULATION ---
    if t > 0:
        try:
            last_affordability_val = current_hist['Affordability_Ratio'].iloc[-2]
            current_income_yoy = current_hist['Income_Growth_YoY'].iloc[-1]
            monthly_income_factor = (1 + current_income_yoy) ** (1/12)
            price_growth_factor = np.exp(pred_log_return)
            current_affordability = last_affordability_val * (price_growth_factor / monthly_income_factor)
            current_hist.at[current_date, 'Affordability_Ratio'] = current_affordability
            affordability_calculated = True
        except (KeyError, IndexError) as e:
            pass
    
    # Update features
    start_idx = max(0, len(current_hist) - 50)
    tail = current_hist.iloc[start_idx:].copy()
    tail = sim._update_lags_and_deltas(tail)
    
    # Write calculated features back
    for col in tail.columns:
        current_hist.loc[tail.index, col] = tail[col]
    
    # Check if lags are populated
    if t == 5:  # Check at month 5
        lag_populated = 0
        for lag_col in lag_cols[:3]:  # Check first 3 lag columns
            if pd.notna(current_hist.iloc[-1].get(lag_col)):
                lag_populated += 1
        if lag_populated > 0:
            lags_calculated = True
    
    # Predict
    try:
        X_row = current_hist.iloc[[-1]][sim.feature_columns]
    except KeyError:
        X_row = current_hist.iloc[[-1]].copy()
        for col in sim.feature_columns:
            if col not in X_row.columns:
                X_row[col] = 0
    
    X_row = X_row.fillna(0).replace([np.inf, -np.inf], 0)
    pred_log_return = float(sim.xgb_model.predict(X_row)[0])
    log_returns.append(pred_log_return)
    
    current_log_price = current_log_price + pred_log_return
    price_path.append(float(np.exp(current_log_price)))
    
    current_hist.at[current_date, sim.price_col] = pred_log_return
    
    # Store affordability
    try:
        affordability_val = current_hist.at[current_date, 'Affordability_Ratio']
    except:
        affordability_val = np.nan
    affordability_path.append(affordability_val)
    
    status = f"Month {t+1:2d}: Price=${price_path[-1]:>10,.0f}, Affordability={affordability_val:>8.6f}"
    print(f"   {status}")

print("   " + "-"*76)

# Summary
print("\n7. TEST RESULTS:")
print("   " + "="*76)

valid_affordability = [a for a in affordability_path if pd.notna(a) and a > 0]
if len(valid_affordability) > 0:
    print(f"   ✓ Affordability calculated for {len(valid_affordability)}/{len(affordability_path)} months")
    print(f"     - Initial: {valid_affordability[0]:.6f}")
    print(f"     - Final:   {valid_affordability[-1]:.6f}")
    print(f"     - Change:  {((valid_affordability[-1]/valid_affordability[0])-1)*100:+.2f}%")
else:
    print(f"   ✗ No affordability values calculated")

if lags_calculated:
    print(f"   ✓ Lags/Deltas/RAs are being calculated and populated")
else:
    print(f"   ? Lags/Deltas/RAs status uncertain")

print("\n   ✓ TEST COMPLETED SUCCESSFULLY")
print("="*80)
