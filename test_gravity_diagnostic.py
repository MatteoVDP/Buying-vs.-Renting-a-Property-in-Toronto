#!/usr/bin/env python3
"""
Detailed diagnostic of the gravity override in action during a real forecast.
This creates a minimal mock scenario to test the logic without full SARIMAX fitting.
"""

import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

# Create mock data to simulate the forecast loop
print("="*80)
print("GRAVITY OVERRIDE DIAGNOSTIC: Simulating forecast_price loop")
print("="*80)

# Simulation parameters (same as in production)
historical_max_affordability = 10.5  # From actual data
sentiment_shock_mean = 0.012         # From simulator config
sentiment_shock_std = 0.05
sentiment_mean_reversion = 0.95

# Starting conditions
previous_simulated_affordability = 10.2  # Start a bit below max
sentiment_score = 0.0
current_log_price = np.log(1090326.0)

print(f"\nStarting conditions:")
print(f"  - Historical Max Affordability: {historical_max_affordability:.2f}")
print(f"  - Initial Affordability: {previous_simulated_affordability:.2f}")
print(f"  - Starting Price: ${np.exp(current_log_price):,.0f}")

# Simulate 24 months
print(f"\n\nRunning 24-month simulation with forwarding-looking gravity override:\n")
print(f"{'Month':<8} {'Aff(start)':<12} {'XGBpred':<10} {'Sentiment':<10} {'PredReturn':<12} {'Override?':<10} {'FinalReturn':<12} {'Aff(end)':<12} {'Price':<15}")
print("-" * 130)

results = []

for t in range(24):
    # Simulate XGBoost prediction (random between -2% and +5%)
    rational_log_return = np.random.uniform(-0.02, 0.05)
    
    # Sentiment accumulation
    monthly_shock = np.random.normal(sentiment_shock_mean, sentiment_shock_std)
    sentiment_score = (sentiment_score * sentiment_mean_reversion) + monthly_shock
    
    # Initial prediction (before override)
    initial_pred_log_return = rational_log_return + sentiment_score
    pred_log_return = initial_pred_log_return
    
    # Get income growth (simulate at ~2% YoY)
    current_income_yoy = np.random.uniform(0.015, 0.035)
    monthly_income_factor = (1 + current_income_yoy) ** (1/12)
    
    # ===== FORWARD-LOOKING GRAVITY CHECK (THE FIX) =====
    price_growth_factor = np.exp(pred_log_return)
    projected_affordability = previous_simulated_affordability * (price_growth_factor / monthly_income_factor)
    
    override_applied = False
    if projected_affordability > historical_max_affordability:
        # Calculate corrective return
        target_price_factor = (historical_max_affordability / previous_simulated_affordability) * monthly_income_factor
        target_log_return = np.log(target_price_factor)
        penalty_multiplier = 0.8
        pred_log_return = target_log_return * penalty_multiplier
        pred_log_return = min(pred_log_return, -0.01)
        override_applied = True
    elif projected_affordability > (historical_max_affordability * 0.95):
        pred_log_return = min(0.0, pred_log_return)
        override_applied = (pred_log_return < initial_pred_log_return)
    
    # Update price
    current_log_price = current_log_price + pred_log_return
    current_price = np.exp(current_log_price)
    
    # Calculate new affordability
    price_growth_factor = np.exp(pred_log_return)
    current_affordability = previous_simulated_affordability * (price_growth_factor / monthly_income_factor)
    
    # Track for next iteration
    previous_simulated_affordability = current_affordability
    
    results.append({
        'month': t + 1,
        'aff_start': previous_simulated_affordability * (1 / price_growth_factor) * monthly_income_factor,  # Reverse to get starting aff
        'aff_end': current_affordability,
        'price': current_price,
        'return': pred_log_return,
        'override': override_applied
    })
    
    # Print row
    aff_start_display = previous_simulated_affordability / price_growth_factor * monthly_income_factor
    print(f"{t+1:<8} {aff_start_display:<12.2f} {rational_log_return*100:<10.2f}% {sentiment_score*100:<10.2f}% {initial_pred_log_return*100:<12.2f}% {'YES' if override_applied else 'NO':<10} {pred_log_return*100:<12.2f}% {current_affordability:<12.2f} ${current_price:<14,.0f}")

print("\n" + "="*80)
print("KEY OBSERVATIONS:")
print("="*80)

df = pd.DataFrame(results)
max_aff = df['aff_end'].max()
final_price = df['price'].iloc[-1]
initial_price = 1090326.0

print(f"\n1. Maximum Affordability Reached: {max_aff:.2f}")
print(f"   - Threshold: {historical_max_affordability:.2f}")
print(f"   - Exceeded? {max_aff > historical_max_affordability}")

print(f"\n2. Final Price (24 months): ${final_price:,.0f}")
print(f"   - Starting Price: ${initial_price:,.0f}")
print(f"   - Total Change: {100*(final_price/initial_price - 1):.1f}%")
print(f"   - Reasonable? {'YES' if final_price < initial_price * 1.5 else 'NO (explosion detected!)'}")

print(f"\n3. Gravity Override Activations:")
override_count = df['override'].sum()
print(f"   - Times override triggered: {override_count} / {len(df)}")
if override_count == 0:
    print(f"     ⚠️  NO OVERRIDES TRIGGERED - affordability stayed below threshold")
else:
    print(f"     ✓ Override working - it constrained affordability growth")

print(f"\n4. Affordability Trend:")
print(f"   - Start: {df['aff_end'].iloc[0]:.2f}")
print(f"   - End: {df['aff_end'].iloc[-1]:.2f}")
print(f"   - Peak: {df['aff_end'].max():.2f}")
if df['aff_end'].iloc[-1] > df['aff_end'].iloc[0]:
    print(f"   - Direction: RISING (this would cause explosion!)")
else:
    print(f"   - Direction: FALLING or STABLE (controlled by gravity)")

print("\n" + "="*80)
if override_count > 0:
    print("✅ GRAVITY OVERRIDE IS WORKING")
else:
    print("⚠️  Gravity override not triggered - affordability stayed low")
print("="*80)
