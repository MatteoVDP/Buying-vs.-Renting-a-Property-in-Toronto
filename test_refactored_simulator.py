#!/usr/bin/env python3
"""
Quick test to verify the refactored MarketSimulator works with new processed_data.csv
"""
import sys
sys.path.insert(0, '/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/scripts')

import pandas as pd
import numpy as np
from market_simulator import MarketSimulator

# Load the new processed data
print("Loading processed_data.csv...")
df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")

# Initialize simulator
print("\n" + "="*60)
print("Initializing MarketSimulator...")
sim = MarketSimulator(df)
print("✓ Simulator initialized")

# Fit the model
print("\n" + "="*60)
print("Fitting XGBoost model...")
try:
    sim.fit(df)
    print(f"✓ Model fit successfully with {len(sim.feature_columns)} features")
    print(f"Feature columns (first 10): {sim.feature_columns[:10]}")
except Exception as e:
    print(f"✗ Error during fit: {e}")
    raise

# Test feature engineering
print("\n" + "="*60)
print("Testing _update_lags_and_deltas...")
try:
    test_df = df.iloc[:50].copy()
    test_df_engineered = sim._update_lags_and_deltas(test_df)
    new_cols = set(test_df_engineered.columns) - set(test_df.columns)
    print(f"✓ Generated {len(new_cols)} new feature columns")
    print(f"Sample new columns: {list(new_cols)[:10]}")
except Exception as e:
    print(f"✗ Error in feature engineering: {e}")
    raise

print("\n" + "="*60)
print("✓ All tests passed! Refactored simulator is working correctly.")
