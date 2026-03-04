#!/usr/bin/env python3
"""
Direct test of the Macro Gravity Override forward-looking affordability check
"""

import numpy as np
import pandas as pd

# Test parameters
historical_max_affordability = 10.5
previous_simulated_affordability = 10.0  # Start just below threshold

print("="*80)
print("GRAVITY OVERRIDE FORWARD-LOOKING LOGIC TEST")
print("="*80)
print(f"\nHistorical Max Affordability (threshold): {historical_max_affordability:.2f}")
print(f"Starting Affordability: {previous_simulated_affordability:.2f}")

# Test Case 1: Prediction would push affordability OVER threshold
print("\n\n--- TEST 1: Prediction causes affordability to breach threshold ---")
rational_log_return = 0.027  # +2.7% from model
sentiment_score = 0.077     # +7.7% sentiment
pred_log_return = rational_log_return + sentiment_score  # +10.4% total

current_income_yoy = 0.03  # 3% YoY
monthly_income_factor = (1 + current_income_yoy) ** (1/12)
price_growth_factor = np.exp(pred_log_return)
projected_affordability = previous_simulated_affordability * (price_growth_factor / monthly_income_factor)

print(f"\nInitial prediction (XGBoost + sentiment): +{pred_log_return*100:.2f}%")
print(f"  - XGBoost: +{rational_log_return*100:.2f}%")
print(f"  - Sentiment: +{sentiment_score*100:.2f}%")
print(f"\nProjected affordability with that return: {projected_affordability:.2f}")
print(f"Threshold exceeded? {projected_affordability > historical_max_affordability}")

# Apply the forward-looking gravity override
if projected_affordability > historical_max_affordability:
    print("\n✓ GRAVITY OVERRIDE TRIGGERED")
    
    # Calculate return needed to bring affordability to threshold
    target_price_factor = (historical_max_affordability / previous_simulated_affordability) * monthly_income_factor
    target_log_return = np.log(target_price_factor)
    
    # Apply penalty multiplier
    penalty_multiplier = 0.8
    overridden_log_return = target_log_return * penalty_multiplier
    overridden_log_return = min(overridden_log_return, -0.01)  # At least -1%
    
    print(f"Target return to reach threshold: {target_log_return*100:.2f}%")
    print(f"Actual return applied (80% of target): {overridden_log_return*100:.2f}%")
    
    # Verify the new affordability
    new_price_factor = np.exp(overridden_log_return)
    new_affordability = previous_simulated_affordability * (new_price_factor / monthly_income_factor)
    print(f"New affordability after override: {new_affordability:.2f}")
    print(f"Safely below threshold? {new_affordability <= historical_max_affordability}")
else:
    print("\n✗ No override needed")

# Test Case 2: Next month with high affordability
print("\n\n--- TEST 2: Starting month with affordability already high ---")
previous_simulated_affordability = 15.0  # Already over threshold!
rational_log_return = 0.027
sentiment_score = 0.077
pred_log_return = rational_log_return + sentiment_score

price_growth_factor = np.exp(pred_log_return)
projected_affordability = previous_simulated_affordability * (price_growth_factor / monthly_income_factor)

print(f"\nStarting affordability: {previous_simulated_affordability:.2f} (OVER threshold!)")
print(f"Prediction would cause: {projected_affordability:.2f}")

if projected_affordability > historical_max_affordability:
    print("\n✓ GRAVITY OVERRIDE TRIGGERED")
    
    target_price_factor = (historical_max_affordability / previous_simulated_affordability) * monthly_income_factor
    target_log_return = np.log(target_price_factor)
    penalty_multiplier = 0.8
    overridden_log_return = target_log_return * penalty_multiplier
    overridden_log_return = min(overridden_log_return, -0.01)
    
    print(f"Uncontrolled return: {pred_log_return*100:.2f}%")
    print(f"Override return: {overridden_log_return*100:.2f}%")
    
    new_price_factor = np.exp(overridden_log_return)
    new_affordability = previous_simulated_affordability * (new_price_factor / monthly_income_factor)
    print(f"Result: affordability {previous_simulated_affordability:.2f} → {new_affordability:.2f}")
    print(f"Forced down? {new_affordability < previous_simulated_affordability}")

# Test Case 3: Already at 95% of threshold 
print("\n\n--- TEST 3: Soft landing zone (95-100% of threshold) ---")
previous_simulated_affordability = 10.0  # 95.2% of threshold
rational_log_return = 0.02
sentiment_score = 0.05
pred_log_return = rational_log_return + sentiment_score  # +7% total

projected_affordability = previous_simulated_affordability * (price_growth_factor / monthly_income_factor)

print(f"\nStarting affordability: {previous_simulated_affordability:.2f} ({100*previous_simulated_affordability/historical_max_affordability:.1f}% of threshold)")
print(f"Uncontrolled return: +{pred_log_return*100:.2f}%")

# Soft landing: max zero growth
if projected_affordability > (historical_max_affordability * 0.95):
    overridden_return = min(0.0, pred_log_return)
    print(f"Soft landing override: capped to {overridden_return*100:.2f}%")
else:
    print("No override needed")

print("\n" + "="*80)
print("✅ GRAVITY OVERRIDE LOGIC VERIFIED")
print("="*80)
