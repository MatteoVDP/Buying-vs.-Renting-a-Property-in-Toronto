"""
25-Year Market Forecast: March 2025 to March 2050
================================================
Runs the market simulator to generate a complete price forecast
with all exogenous variables filled for the 25-year period.
Outputs: CSV with complete data + visualization of price path.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator


def run_25_year_forecast():
    """Run single 25-year (300-month) forecast and save results."""
    
    print("="*80)
    print("25-YEAR MARKET FORECAST: MARCH 2025 - MARCH 2050")
    print("="*80)
    
    # 1. LOAD DATA
    print("\n1. Loading historical market data...")
    df = pd.read_csv('data/processed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.asfreq('MS')
    df = df.ffill()
    
    last_date = df.index.max()
    print(f"   Historical data: {len(df)} rows")
    print(f"   Date range: {df.index.min().strftime('%B %Y')} to {last_date.strftime('%B %Y')}")
    
    # 2. INITIALIZE SIMULATOR
    print("\n2. Initializing MarketSimulator with seed=42...")
    sim = MarketSimulator(df, seed=42)
    
    # 3. FIT MODELS
    print("\n3. Fitting ARIMA/SARIMAX/XGBoost models on historical data...")
    import sys
    sys.stdout.flush()
    sim.fit()
    sys.stdout.flush()
    print("   ✓ Models fitted successfully")
    sys.stdout.flush()
    
    # 4. GENERATE 25-YEAR FORECAST (300 months)
    print("\n4. Generating 25-year (300-month) forecast...")
    steps = 300
    print(f"   Running 1 Monte Carlo iteration x {steps} months...")

    # Use March 2025 market price as the base for log return accumulation
    start_market_price = 1090326.0
    sim.start_market_price = float(start_market_price)

    # Generate future dates
    future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

    # 5. BUILD EXOGENOUS DATA FOR FORECAST PERIOD
    print("\n5. Generating exogenous variables for forecast period...")
    sim_exog = sim.simulate_exogenous(steps=steps)

    print(f"   ✓ Forecast horizon: {len(future_index)} months")
    print(f"   Forecast period: {future_index[0].strftime('%B %Y')} to {future_index[-1].strftime('%B %Y')}")
    
    # Create output DataFrame with exogenous variables
    forecast_df = sim_exog.copy()
    forecast_df.index = future_index
    
    # Predict log returns and convert to real prices from the base price
    print("\n6. Predicting Log_Return_MoM and converting to real prices...")
    base_hist = sim.df.copy()
    current_hist = base_hist.copy()
    
    log_returns = []
    price_path = []
    affordability_path = []
    current_log_price = float(np.log(start_market_price))
    pred_log_return = 0  # Initialize for first iteration
    
    for t in range(steps):
        current_date = future_index[t]
        sim_row = sim_exog.iloc[[t]]
        
        # Ensure sim_row has all columns from current_hist
        for col in current_hist.columns:
            if col not in sim_row.columns:
                sim_row[col] = np.nan
        
        current_hist = pd.concat([current_hist, sim_row], axis=0)
        current_hist = current_hist.ffill()
        
        # --- DYNAMIC AFFORDABILITY RECALCULATION (before lags/deltas) ---
        if t > 0:  # Can only calculate after first step
            try:
                # 1. Get last month's known affordability ratio
                last_affordability_val = current_hist['Affordability_Ratio'].iloc[-2]  # -2 because we just added a row
                
                # 2. Get the current simulated Income Growth (YoY) and de-annualize it to a MoM factor
                current_income_yoy = current_hist['Income_Growth_YoY'].iloc[-1]
                monthly_income_factor = (1 + current_income_yoy) ** (1/12)
                
                # 3. Convert XGBoost's predicted log return into a simple price growth factor
                price_growth_factor = np.exp(pred_log_return)
                
                # 4. Calculate the new ratio 
                current_affordability = last_affordability_val * (price_growth_factor / monthly_income_factor)
                
                # 5. Write it back to the history dataframe immediately
                current_hist.at[current_date, 'Affordability_Ratio'] = current_affordability
            except (KeyError, IndexError) as e:
                # Fallback: use last known value
                current_affordability = current_hist['Affordability_Ratio'].dropna().iloc[-1]
                current_hist.at[current_date, 'Affordability_Ratio'] = current_affordability
        
        # Update features (Calculates Lags, Deltas, and Rolling Averages)
        # Use wider lookback window to ensure enough history for all lags
        start_idx = max(0, len(current_hist) - 50)
        tail = current_hist.iloc[start_idx:].copy()
        tail = sim._update_lags_and_deltas(tail)
        
        # Write all calculated features back to current_hist
        for col in tail.columns:
            current_hist.loc[tail.index, col] = tail[col]
        
        # Extract the row to predict (the very last one)
        try:
            X_row = current_hist.iloc[[-1]][sim.feature_columns]
        except KeyError:
            # Some feature columns may not exist, handle gracefully
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
        
        # Store current affordability for output
        try:
            affordability_val = current_hist.at[current_date, 'Affordability_Ratio']
        except:
            affordability_val = np.nan
        affordability_path.append(affordability_val)

    # Extract forecast period from current_hist (which has all calculated lags/deltas/RAs)
    print("\n   Extracting calculated features from forecast period...")
    forecast_df = current_hist.loc[future_index].copy()
    
    # Ensure Market_Price and Log_Price are in the dataframe
    forecast_df['Market_Price'] = price_path
    forecast_df['Log_Price'] = np.log(forecast_df['Market_Price'])
    
    # Calculate price statistics
    forecast_df['Price_Change_MoM_%'] = forecast_df['Market_Price'].pct_change() * 100
    forecast_df['Price_Change_YoY_%'] = forecast_df['Market_Price'].pct_change(12) * 100
    
    # Calculate affordability statistics
    if 'Affordability_Ratio' in forecast_df.columns:
        forecast_df['Affordability_Change_MoM_%'] = forecast_df['Affordability_Ratio'].pct_change() * 100
        forecast_df['Affordability_Change_YoY_%'] = forecast_df['Affordability_Ratio'].pct_change(12) * 100
    
    # Count how many lag/delta/RA columns are populated
    lag_cols = [c for c in forecast_df.columns if '_lag_' in c]
    delta_cols = [c for c in forecast_df.columns if '_delta_' in c]
    ra_cols = [c for c in forecast_df.columns if '_RA_' in c]
    print(f"   Forecast dataframe has {len(lag_cols)} lag, {len(delta_cols)} delta, {len(ra_cols)} RA columns")
    
    # 7. COMBINE HISTORICAL + FORECAST
    print("\n7. Combining historical and forecast data...")
    
    # Add missing columns to historical data (they won't exist yet)
    for col in ['Market_Price', 'Log_Price', 'Price_Change_MoM_%', 'Price_Change_YoY_%']:
        if col not in df.columns:
            df[col] = np.nan
    
    # Compute historical prices (if not available)
    if df['Market_Price'].isna().all():
        # Use Log_Return_MoM to reconstruct prices (if available)
        # Otherwise mark as historical
        df['Market_Price'] = np.nan
        df['Log_Price'] = np.nan
    
    # Mark data source
    df['Data_Source'] = 'Historical'
    forecast_df['Data_Source'] = 'Forecast'
    
    # Concatenate
    combined_df = pd.concat([df, forecast_df])
    combined_df = combined_df.sort_index()
    
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Total columns: {len(combined_df.columns)}")
    print(f"   Date range: {combined_df.index.min().strftime('%B %Y')} to {combined_df.index.max().strftime('%B %Y')}")
    
    # 8. SAVE TO CSV
    output_csv = 'results/forecast_25_year_march2050.csv'
    print(f"\n7. Saving complete forecast to {output_csv}...")
    combined_df.to_csv(output_csv)
    print(f"   ✓ Saved {len(combined_df)} rows × {len(combined_df.columns)} columns")
    
    # 9. CREATE VISUALIZATION
    print("\n8. Creating price path visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- MAIN PRICE PATH ---
    ax = axes[0]
    
    # Historical prices (if available)
    hist_mask = combined_df['Data_Source'] == 'Historical'
    if hist_mask.any() and combined_df[hist_mask]['Market_Price'].notna().any():
        hist_prices = combined_df[hist_mask]
        ax.plot(hist_prices.index, hist_prices['Market_Price'], 
               color='navy', linewidth=2.5, label='Historical', alpha=0.9)
    
    # Forecast prices
    forecast_mask = combined_df['Data_Source'] == 'Forecast'
    forecast_data = combined_df[forecast_mask]
    ax.plot(forecast_data.index, forecast_data['Market_Price'], 
           color='crimson', linewidth=2.5, label='Forecast (to March 2050)', 
           linestyle='--', alpha=0.8)
    
    # Add vertical line at forecast start
    ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Forecast Start')
    
    # Formatting
    ax.set_title('25-Year Market Price Forecast: March 2025 - March 2050', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Market Price ($)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # --- MONTHLY % CHANGE ---
    ax2 = axes[1]
    
    forecast_changes = forecast_data['Price_Change_MoM_%'].dropna()
    ax2.bar(forecast_data.index[1:], forecast_changes, 
           color=['green' if x > 0 else 'red' for x in forecast_changes],
           alpha=0.6, width=20, label='Monthly % Change')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_title('Monthly Price Changes (%)', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('% Change', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Format x-axis dates
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_fig = 'results/forecast_25_year_march2050.png'
    print(f"   Saving visualization to {output_fig}...")
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved")
    
    # 10. SUMMARY STATISTICS
    print("\n10. Forecast Summary Statistics:")
    print("   " + "="*76)
    
    forecast_prices_only = forecast_data['Market_Price'].dropna()
    
    if len(forecast_prices_only) > 0:
        initial_price = forecast_prices_only.iloc[0]
        final_price = forecast_prices_only.iloc[-1]
        min_price = forecast_prices_only.min()
        max_price = forecast_prices_only.max()
        mean_price = forecast_prices_only.mean()
        
        total_change_pct = ((final_price / initial_price) - 1) * 100
        total_change_avg_annual = ((final_price / initial_price) ** (1/25) - 1) * 100
        
        print(f"   Initial Price (Start of Forecast): ${initial_price:,.2f}")
        print(f"   Final Price (March 2050):          ${final_price:,.2f}")
        print(f"   Lowest Price:                      ${min_price:,.2f}")
        print(f"   Highest Price:                     ${max_price:,.2f}")
        print(f"   Mean Price:                        ${mean_price:,.2f}")
        print(f"\n   Total 25-Year Change:              {total_change_pct:+.2f}%")
        print(f"   Average Annual Growth Rate:        {total_change_avg_annual:+.2f}%")
        print(f"\n   Monthly Returns - Mean:            {forecast_data['Price_Change_MoM_%'].mean():+.3f}%")
        print(f"   Monthly Returns - Std Dev:         {forecast_data['Price_Change_MoM_%'].std():.3f}%")
        print(f"   Monthly Returns - Min:             {forecast_data['Price_Change_MoM_%'].min():+.3f}%")
        print(f"   Monthly Returns - Max:             {forecast_data['Price_Change_MoM_%'].max():+.3f}%")
    
    # Affordability Statistics
    print("\n   " + "-"*76)
    print("   AFFORDABILITY RATIO STATISTICS:")
    print("   " + "-"*76)
    
    if 'Affordability_Ratio' in forecast_data.columns:
        affordability_valid = forecast_data['Affordability_Ratio'].dropna()
        if len(affordability_valid) > 0:
            initial_afford = affordability_valid.iloc[0]
            final_afford = affordability_valid.iloc[-1]
            min_afford = affordability_valid.min()
            max_afford = affordability_valid.max()
            mean_afford = affordability_valid.mean()
            
            afford_change_pct = ((final_afford / initial_afford) - 1) * 100 if initial_afford != 0 else 0
            
            print(f"   Initial Affordability:             {initial_afford:,.4f}")
            print(f"   Final Affordability (March 2050):  {final_afford:,.4f}")
            print(f"   Lowest Affordability:              {min_afford:,.4f}")
            print(f"   Highest Affordability:             {max_afford:,.4f}")
            print(f"   Mean Affordability:                {mean_afford:,.4f}")
            print(f"   25-Year Change:                    {afford_change_pct:+.2f}%")
            print(f"   Monthly Change - Mean:             {forecast_data['Affordability_Change_MoM_%'].mean():+.3f}%")
            print(f"   Monthly Change - Std Dev:          {forecast_data['Affordability_Change_MoM_%'].std():.3f}%")
        else:
            print("   ⚠️  No affordability values found in forecast")
    else:
        print("   ⚠️  Affordability_Ratio column not in forecast data")
    
    # Feature Engineering Diagnostics
    print("\n   " + "-"*76)
    print("   FEATURE ENGINEERING DIAGNOSTICS:")
    print("   " + "-"*76)
    
    # Check for lag/delta/RA columns
    lag_cols = [c for c in combined_df.columns if '_lag_' in c]
    delta_cols = [c for c in combined_df.columns if '_delta_' in c]
    ra_cols = [c for c in combined_df.columns if '_RA_' in c]
    
    print(f"   Lag columns defined: {len(lag_cols)}")
    print(f"   Delta columns defined: {len(delta_cols)}")
    print(f"   Rolling Average columns defined: {len(ra_cols)}")
    
    # Sample a few rows from forecast to check if lags/deltas are populated
    forecast_with_features = combined_df[combined_df['Data_Source'] == 'Forecast'].copy()
    if len(forecast_with_features) > 24:
        sample_row = forecast_with_features.iloc[24]  # Check row after 24 months
        populated_lags = sum(1 for c in lag_cols if pd.notna(sample_row.get(c)))
        populated_deltas = sum(1 for c in delta_cols if pd.notna(sample_row.get(c)))
        populated_ras = sum(1 for c in ra_cols if pd.notna(sample_row.get(c)))
        
        print(f"\n   At Month 24 of forecast:")
        print(f"     - Lags populated: {populated_lags}/{len(lag_cols)}")
        print(f"     - Deltas populated: {populated_deltas}/{len(delta_cols)}")
        print(f"     - Rolling Averages populated: {populated_ras}/{len(ra_cols)}")
    
    print("   " + "="*76)
    
    # 11. COMPLETION
    print("\n" + "="*80)
    print("FORECAST GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  1. Complete CSV: {output_csv}")
    print(f"     - {len(combined_df)} rows (historical + forecast)")
    print(f"     - All exogenous variables filled for forecast period")
    print(f"     - Price projections from March 2025 to March 2050")
    print(f"\n  2. Visualization: {output_fig}")
    print(f"     - Monthly price path with confidence visual")
    print(f"     - Monthly percent changes")
    print(f"\nRecommended next steps:")
    print(f"  - Review CSV in Excel for detailed month-by-month data")
    print(f"  - Open PNG to visualize the forecast trajectory")
    print(f"  - Compare with multiple iterations if desired")
    

if __name__ == '__main__':
    run_25_year_forecast()
