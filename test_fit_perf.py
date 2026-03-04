#!/usr/bin/env python3
"""
Test if the fit() function has performance issues
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from market_simulator import MarketSimulator

# Load data
data_path = Path(__file__).parent / "data" / "processed_data.csv"
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print("Testing fit() performance...")

# Split: use first 80% for training
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Initialize
sim = MarketSimulator(
    df=df.copy(),
    seed=42
)

print("Starting fit()...")
import time
start = time.time()

try:
    sim.fit(train_df)
    elapsed = time.time() - start
    print(f"✓ fit() completed in {elapsed:.2f} seconds")
    print(f"  historical_max_affordability = {sim.historical_max_affordability:.4f}")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ fit() failed after {elapsed:.2f} seconds")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
