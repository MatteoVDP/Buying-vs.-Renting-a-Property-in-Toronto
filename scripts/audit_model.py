"""
Transparency Diagnostic Script for MarketSimulator
====================================================
Runs a SINGLE Monte Carlo iteration and captures the FULL month-by-month state
of all variables, reconstructed levels, and dynamic features for manual inspection.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator


def audit_single_iteration():
    """Run one complete simulation iteration and construct full audit DataFrame."""
    
    print("="*80)
    print("MARKET SIMULATOR TRANSPARENCY AUDIT")
    print("="*80)
    
    # 1. LOAD DATA
    print("\n1. Loading data...")
    df = pd.read_csv('data/processed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.asfreq('MS')
    df = df.ffill()
    print(f"   Historical data shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # 2. INITIALIZE SIMULATOR WITH FIXED SEED
    print("\n2. Initializing MarketSimulator with seed=42...")
    sim = MarketSimulator(df, seed=42)
    
    # 3. FIT MODELS
    print("\n3. Fitting models...")
    sim.fit()
    
    # 4. RUN SINGLE ITERATION MANUALLY (to collect state at each step)
    print("\n4. Running single 300-month iteration with full state capture...")
    steps = 300
    last_date = sim.df.index.max()
    future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    
    # Initialize collection variables
    audit_rows = []
    base_hist = sim.df.copy()
    
    # Generate exogenous world (one time, same for all iterations)
    print("   Generating exogenous variables (Tiers 1-3)...")
    sim_world = sim.simulate_exogenous(steps=steps)
    
    current_hist = base_hist.copy()
    
    # Initialize cumulative log price from starting market price
    current_log_price = np.log(sim.start_market_price)
    
    # Step through each month
    for t in range(steps):
        if (t + 1) % 50 == 0:
            print(f"   Processing month {t+1}/{steps}...")
        
        current_date = sim_world.index[t]
        
        # Get simulated exogenous row
        sim_row = sim_world.iloc[[t]].copy()
        
        # Append to history
        current_hist = pd.concat([current_hist, sim_row])
        
        # Update lags/deltas
        tail = current_hist.iloc[-24:].copy()
        tail = sim._update_lags_and_deltas(tail)
        
        # Predict monthly log return
        X_row = tail.iloc[[-1]][sim.feature_columns]
        X_row = X_row.fillna(0).replace([np.inf, -np.inf], 0)
        pred_log_return = float(sim.xgb_model.predict(X_row)[0])
        
        # Write monthly return back to history for next month's lags
        current_hist.at[current_date, sim.price_col] = pred_log_return
        
        # Build audit row
        audit_row = {}
        audit_row['date'] = current_date
        audit_row['step'] = t + 1
        
        # Tier 1: Growth Rates
        for v in sim.tier1_vars:
            if v in sim_world.columns:
                audit_row[v] = sim_world.loc[current_date, v]
        
        # Tier 1: Reconstructed Levels
        for rate_col, level_col in sim.growth_to_level_map.items():
            if level_col in sim_world.columns:
                audit_row[level_col] = sim_world.loc[current_date, level_col]
        
        # Tier 2 & 3: All simulated variables
        for v in sim.tier2_vars + sim.tier3_vars:
            if v in sim_world.columns:
                audit_row[v] = sim_world.loc[current_date, v]
        
        # Tier 4: Write predicted monthly return and cumulative log price
        audit_row['Log_Return_MoM'] = pred_log_return
        
        # Update cumulative log price by accumulating monthly returns
        current_log_price = current_log_price + pred_log_return
        audit_row['Log_Price'] = current_log_price
        audit_row['Market_Price'] = np.exp(current_log_price)
        
        # Include feature lags/deltas every month (always include these columns)
        last_with_features = tail.iloc[[-1]]
        for col in sim.feature_columns:
            # Only include lags and deltas, not raw values
            if '_lag_' in col or '_delta_' in col:
                value = last_with_features[col].iloc[0] if col in last_with_features.columns else np.nan
                audit_row[col] = value
        
        audit_rows.append(audit_row)
    
    # 5. CONSTRUCT AUDIT DATAFRAME
    print("\n5. Constructing audit DataFrame...")
    audit_df = pd.DataFrame(audit_rows)
    audit_df.set_index('date', inplace=True)
    
    # Remove entirely empty columns (all NaN) before reporting
    completely_empty = audit_df.columns[audit_df.isna().all()]
    if len(completely_empty) > 0:
        print(f"   Removing {len(completely_empty)} entirely-empty columns...")
        audit_df = audit_df.drop(columns=completely_empty)
    
    print(f"   Audit DataFrame shape: {audit_df.shape}")
    print(f"   Columns: {len(audit_df.columns)}")
    
    # 6. SANITY CHECKS
    print("\n6. Running sanity checks...")
    
    # Check for NaNs
    nan_counts = audit_df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"   WARNING: Found NaNs in {len(nan_cols)} columns:")
        for col, count in nan_cols.items():
            print(f"      {col}: {count} NaNs")
    else:
        print("   ✓ No NaNs detected")
    
    # Check for Infs
    inf_mask = np.isinf(audit_df.select_dtypes(include=[np.number])).sum()
    inf_cols = inf_mask[inf_mask > 0]
    if len(inf_cols) > 0:
        print(f"   WARNING: Found Infs in {len(inf_cols)} columns:")
        for col, count in inf_cols.items():
            print(f"      {col}: {count} Infs")
    else:
        print("   ✓ No Infinities detected")
    
    # Verify price monotonicity (optional: just check for extreme jumps)
    price_col = 'Market_Price'
    if price_col in audit_df.columns:
        price_pct_changes = audit_df[price_col].pct_change().abs()
        max_jump = price_pct_changes.max()
        print(f"   Market Price - Max monthly change: {max_jump*100:.2f}%")
        if max_jump > 0.20:
            print(f"      WARNING: Large jump detected (>20%)")
        else:
            print(f"      ✓ Price changes appear reasonable")
    
    # Verify level reconstruction (spot check GDP)
    gdp_rate_col = 'GDP_Growth_YoY'
    gdp_level_col = 'national_gdp_real,_seasonally_adjusted'
    if gdp_rate_col in audit_df.columns and gdp_level_col in audit_df.columns:
        # Check if levels were reconstructed correctly
        # New = Old * (1 + Rate)
        sample_idx = [10, 50, 100, 200, 299]
        print(f"   GDP Level Reconstruction Check (sample steps):")
        for idx in sample_idx:
            if idx < len(audit_df):
                rate = audit_df[gdp_rate_col].iloc[idx]
                level = audit_df[gdp_level_col].iloc[idx]
                prev_level = audit_df[gdp_level_col].iloc[idx-1] if idx > 0 else None
                if prev_level is not None:
                    expected = prev_level * (1 + rate)
                    error = abs(level - expected) / expected if expected != 0 else 0
                    status = "✓" if error < 0.01 else "✗"
                    print(f"      Step {idx}: Level={level:.2f}, Expected={expected:.2f}, Error={error*100:.4f}% {status}")
    
    # 7. HEAD & TAIL
    print("\n7. HEAD (First 5 simulated months):")
    head_cols = [gdp_rate_col, gdp_level_col, 'variable_mortgage_rate', 'Log_Price', 'Market_Price']
    head_cols = [c for c in head_cols if c in audit_df.columns]
    print(audit_df[head_cols].head())
    
    print("\n8. TAIL (Last 5 simulated months):")
    print(audit_df[head_cols].tail())
    
    # 8. SAVE TO CSV
    output_path = 'results/simulation_audit_trace.csv'
    print(f"\n9. Saving full audit trace to {output_path}...")
    audit_df.to_csv(output_path)
    print(f"   ✓ Saved {len(audit_df)} rows × {len(audit_df.columns)} columns")
    
    # Summary statistics
    print("\n10. Summary Statistics:")
    numeric_cols = audit_df.select_dtypes(include=[np.number]).columns
    summary = audit_df[numeric_cols].describe()
    print(summary.to_string())
    
    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  - Simulation period: March 2025 to March 2050 ({steps} months)")
    print(f"  - Total rows in audit: {len(audit_df)}")
    print(f"  - Total columns: {len(audit_df.columns)}")
    print(f"  - Initial Market Price: ${audit_df['Market_Price'].iloc[0]:,.2f}")
    print(f"  - Final Market Price: ${audit_df['Market_Price'].iloc[-1]:,.2f}")
    print(f"  - Price 25-year change: {((audit_df['Market_Price'].iloc[-1] / audit_df['Market_Price'].iloc[0]) - 1)*100:.2f}%")
    print(f"\nFull trace saved to: {output_path}")
    print("Open this file in Excel or Python for detailed row-by-row inspection.")


if __name__ == '__main__':
    audit_single_iteration()
