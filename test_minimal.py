#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/scripts')
import pandas as pd

# Just test loading data
df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
print("Data loaded: ", len(df), "rows")

# Test import
from market_simulator import MarketSimulator
print("MarketSimulator imported successfully")

# Test initialization
sim = MarketSimulator(df)
print("Simulator initialized")
print("Feature columns will be:", len([c for c in df.columns if c != 'Log_Return_MoM']))
