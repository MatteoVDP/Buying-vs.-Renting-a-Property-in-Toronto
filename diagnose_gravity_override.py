#!/usr/bin/env python3
"""
Diagnostic to check if Macro Gravity Override is working correctly.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('scripts')
from market_simulator import MarketSimulator

print("="*80)
print("DIAGNOSING MACRO GRAVITY OVERRIDE EFFECTIVENESS")
print("="*80)

# Load and fit
df = pd.read_csv('data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS').ffill()

sim = MarketSimulator(df=df, seed=42)
sim.fit()

print(f"\n1. Historical Max Affordability: {sim.historical_max_affordability:.4f}")
print(f"   (90% threshold for soft landing: {sim.historical_max_affordability * 0.9:.4f})")

# Run short forecast and track affordability
print("\n2. Running 60-month forecast and tracking affordability...")
print(f"\n   {'Month':<8} {'Affordability':<15} {'Rational Return':<15} {'Pred Return':<15} {'Applied Override':<15}")
print("   " + "-"*70)

steps = 60
future_index = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

sim_world = sim.simulate_exogenous(steps=steps)
base_hist = df.copy()
current_hist = base_hist.copy()
current_log_price = np.log(sim.start_market_price)
sentiment_score = 0.0

for t in range(steps):
    current_date = future_index[t]
    sim_row = sim_world.iloc[[t]]
    
    for col in current_hist.columns:
        if col not in sim_row.columns:
            sim_row[col] = np.nan
    
    current_hist = pd.concat([current_hist, sim_row], axis=0)
    current_hist = current_hist.ffill()
    
    start_idx = max(0, len(current_hist) - 50)
    tail = current_hist.iloc[start_idx:].copy()
    tail = sim._update_lags_and_deltas(tail)
    
    current_row_features = tail.iloc[-1]
    for col in tail.columns:
        if any(tag in col for tag in ['_lag_', '_delta_', '_RA_', 'month_']):
            if col not in current_hist.columns:
                current_hist[col] = np.nan
            current_hist.at[current_date, col] = current_row_features[col]
    
    try:
        X_row = current_hist.iloc[[-1]][sim.feature_columns]
    except KeyError:
        X_row = current_hist.iloc[[-1]].copy()
        for col in sim.feature_columns:
            if col not in X_row.columns:
                X_row[col] = 0
    
    X_row = X_row.fillna(0).replace([np.inf, -np.inf], 0)
    rational_log_return = float(sim.xgb_model.predict(X_row)[0])
    
    # Check affordability before override
    try:
        current_affordability = current_hist['Affordability_Ratio'].dropna().iloc[-1]
    except IndexError:
        current_affordability = np.nan
    
    rational_before_override = rational_log_return
    
    # Apply override
    try:
        if not np.isnan(current_affordability):
            if current_affordability > sim.historical_max_affordability:
                penalty = (current_affordability - sim.historical_max_affordability) * 0.01
                rational_log_return = min(-penalty, rational_log_return)
            elif current_affordability > (sim.historical_max_affordability * 0.9):
                rational_log_return = min(0.0, rational_log_return)
    except (KeyError, IndexError):
        pass
    
    # Sentiment
    monthly_shock = np.random.normal(sim.sentiment_shock_mean, sim.sentiment_shock_std)
    sentiment_score = (sentiment_score * sim.sentiment_mean_reversion) + monthly_shock
    pred_log_return = rational_log_return + sentiment_score
    
    # Update price
    current_log_price = current_log_price + pred_log_return
    current_hist.at[current_date, sim.price_col] = pred_log_return
    
    # Recalculate affordability
    try:
        last_affordability = current_hist['Affordability_Ratio'].dropna().iloc[-1]
        current_income_yoy = current_hist['Income_Growth_YoY'].iloc[-1]
        monthly_income_factor = (1 + current_income_yoy) ** (1/12)
        price_growth_factor = np.exp(pred_log_return)
        new_affordability = last_affordability * (price_growth_factor / monthly_income_factor)
        current_hist.at[current_date, 'Affordability_Ratio'] = new_affordability
    except (KeyError, IndexError):
        pass
    
    # Diagnostic output every 10 months
    if t % 10 == 0:
        override_applied = rational_before_override != rational_log_return
        print(f"   {t+1:<8} {current_affordability:<15.4f} {rational_before_override:<15.6f} {pred_log_return:<15.6f} {str(override_applied):<15}")

# Final summary
print(f"\n3. Final Analysis:")
final_affordability = current_hist['Affordability_Ratio'].iloc[-1]
print(f"   Starting affordability (from historical): {sim.historical_max_affordability:.4f}")
print(f"   Final affordability after 60 months:      {final_affordability:.4f}")

if final_affordability > sim.historical_max_affordability * 1.5:
    print(f"   ❌ FAILED: Affordability has grown {final_affordability/sim.historical_max_affordability:.2f}x historical max!")
elif final_affordability > sim.historical_max_affordability * 1.1:
    print(f"   ⚠️  PARTIAL: Affordability at {final_affordability/sim.historical_max_affordability:.2f}x, should be constrained")
else:
    print(f"   ✅ SUCCESS: Affordability held near historical levels ({final_affordability/sim.historical_max_affordability:.2f}x)")

print("\n" + "="*80)
