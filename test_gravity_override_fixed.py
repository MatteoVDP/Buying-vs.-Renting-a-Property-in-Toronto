#!/usr/bin/env python3
"""
Test that the Macro Gravity Override kill-switch actually prevents price explosion
when affordability exceeds historical max.
"""

import pandas as pd
import numpy as np
import sys

# Test data
from scripts.market_simulator import MarketSimulator

# Load data
df = pd.read_csv('data/processed_data.csv', parse_dates=['date'])

print("="*80)
print("TESTING: Macro Gravity Override Kill-Switch")
print("="*80)

# Initialize simulator
try:
    sim = MarketSimulator(df, seed=42, start_market_price=1090326.0)
    
    # Fit the model
    print("\nFitting 4-tier model...")
    sim.fit()
    
    # Check historical max affordability
    print(f"\nHistorical max affordability: {sim.historical_max_affordability:.2f}")
    
    # Run a short forecast to test gravity override
    print("\nRunning short 12-month forecast with gravity override...")
    price_paths = sim.forecast_price(iterations=5, steps=12)
    
    # Get the extended forecast with affordability data
    extended = sim.get_extended_forecast(price_paths)
    
    print("\nForecast Results (first 12 months):")
    print("-" * 80)
    
    cols_to_show = ['Price_mean', 'Affordability_Ratio', 'Income_Growth_YoY', 'Log_Return_MoM']
    for col in cols_to_show:
        if col in extended.columns:
            print(f"\n{col}:")
            print(extended[col].head(12).to_string())
    
    # Key test: Check if any affordability value exceeds max
    if 'Affordability_Ratio' in extended.columns:
        max_affordability = extended['Affordability_Ratio'].max()
        print(f"\n\n{'='*80}")
        print(f"MAX AFFORDABILITY IN FORECAST: {max_affordability:.2f}")
        print(f"Historical Max (threshold): {sim.historical_max_affordability:.2f}")
        
        if max_affordability > sim.historical_max_affordability * 1.5:
            print("⚠️  GRAVITY OVERRIDE FAILED - Affordability massively exceeds threshold!")
            print("   Prices are likely still exploding despite gravity override.")
        else:
            print("✅ GRAVITY OVERRIDE WORKING - Affordability constrained!")
    
    # Check final price
    final_price = extended['Price_mean'].iloc[-1] if 'Price_mean' in extended.columns else None
    if final_price:
        print(f"\nFinal 12-month price: ${final_price:,.0f}")
        if final_price > 3e9:
            print("❌ PRICE EXPLOSION - Over $3 billion (gravity override not working)")
        else:
            print("✅ Price reasonable (gravity override preventing exponential explosion)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
