"""
Rolling Walk-Forward Validation Script for MarketSimulator
============================================================
Robust validation strategy: Instead of predicting 100 months in one shot,
predict 12 months at a time, then "walk forward" by retraining on updated data.
This prevents "Anchor Drift" (error compounding over long forecasts).

Architecture:
- Split data into 12-month rolling windows
- For each window: retrain model, predict 12 months, compare to ground truth
- Evaluate on concatenated predictions across all windows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from market_simulator import MarketSimulator
import warnings

warnings.filterwarnings("ignore")

# Configuration
TEST_WINDOW = 12  # Predict 12 months at a time
TOTAL_TEST_MONTHS = 100  # Last 100 months are for testing
ITERATIONS_PER_WINDOW = 20  # 20 Monte Carlo iterations per 12-month window
RESULTS_DIR = "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/results"


def load_and_prepare_data():
    """Load processed data, validate structure, and prepare for rolling validation."""
    print("=" * 80)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(
        "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv"
    )
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date (critical for time series)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"✓ Total span: {len(df)} months (~{len(df) / 12:.1f} years)")
    
    # Calculate split indices
    n_records = len(df)
    test_start_idx = n_records - TOTAL_TEST_MONTHS
    
    print(f"\n✓ Training + Validation split:")
    print(f"  - Training candidate pool: {n_records - TOTAL_TEST_MONTHS} records")
    print(f"  - Testing pool: {TOTAL_TEST_MONTHS} records (last 100 months)")
    print(f"  - Test period: {df.iloc[test_start_idx]['date'].date()} to {df.iloc[-1]['date'].date()}")
    
    return df, test_start_idx


def calculate_start_price(df_slice):
    """
    Extract the actual market price at the END of a training slice.
    This is the "anchor point" - the price the simulator starts from.
    
    Strategy: Use the lag_1 column from the NEXT row (if available),
    or reconstruct from log returns.
    """
    price_col_lag1 = 'market_price_target_average,_detached_single_family_homes_lag_1'
    
    if price_col_lag1 in df_slice.columns:
        # The lag_1 column contains the price from the previous month
        # So for the last row of train, we need to compute what the current price is
        # Price(t) = Price(t-1) * exp(Log_Return(t))
        last_row_idx = len(df_slice) - 1
        if last_row_idx > 0:
            prev_price = df_slice[price_col_lag1].iloc[-1]
            log_return = df_slice['Log_Return_MoM'].iloc[-1]
            if pd.notna(log_return):
                current_price = prev_price * np.exp(log_return)
                return float(current_price)
        # Fallback: use lag_1 directly
        return float(df_slice[price_col_lag1].iloc[-1])
    
    # Fallback: search for any price column
    price_cols = [c for c in df_slice.columns if 'market_price' in c.lower() and 'lag' not in c.lower()]
    if price_cols:
        return float(df_slice[price_cols[0]].dropna().iloc[-1])
    
    raise ValueError("Could not determine start price from data slice")


def extract_actual_prices(test_slice):
    """
    Reconstruct actual observed prices from log returns in the test slice.
    Returns array of prices for the 12-month window.
    """
    log_returns = test_slice['Log_Return_MoM'].fillna(0).values
    # Start from the first price in the slice (already anchored by caller)
    prices = []
    # We need the starting price - comes from previous calculation
    # But here we extract from the slice itself
    
    # Use the lag_1 value from first row as baseline
    price_col_lag1 = 'market_price_target_average,_detached_single_family_homes_lag_1'
    if price_col_lag1 in test_slice.columns and pd.notna(test_slice[price_col_lag1].iloc[0]):
        current_price = float(test_slice[price_col_lag1].iloc[0])
    else:
        current_price = 0  # Will be overridden by caller
    
    for ret in log_returns:
        current_price = current_price * np.exp(ret)
        prices.append(current_price)
    
    return np.array(prices)


def run_rolling_validation():
    """Execute the rolling walk-forward validation."""
    print("\n" + "=" * 80)
    print("STEP 2: Rolling Walk-Forward Validation (12-Month Windows)")
    print("=" * 80)
    
    # Load data
    df, test_start_idx = load_and_prepare_data()
    
    # Storage for all windows
    all_windows = []
    window_num = 0
    
    # Loop: walk forward through test period in 12-month increments
    current_idx = test_start_idx
    
    print(f"\nStarting walk-forward loop...")
    print(f"Number of windows to process: {(len(df) - 12 - test_start_idx) // TEST_WINDOW + 1}")
    
    while current_idx + TEST_WINDOW <= len(df):
        window_num += 1
        window_start_date = df.iloc[current_idx]['date'].date()
        window_end_date = df.iloc[current_idx + TEST_WINDOW - 1]['date'].date()
        
        print(f"\n{'─' * 80}")
        print(f"Window {window_num}: {window_start_date} to {window_end_date}")
        print(f"{'─' * 80}")
        
        # STEP A: Slice Data
        train_df = df.iloc[:current_idx].copy()
        test_df = df.iloc[current_idx:current_idx + TEST_WINDOW].copy()
        
        print(f"  Train records: {len(train_df)} | Test records: {len(test_df)}")
        
        # STEP B: Initialize & Fit
        try:
            # Get actual price at the end of training slice
            start_price = calculate_start_price(train_df)
            print(f"  Anchor price (end of train): ${start_price:,.2f}")
            
            # Initialize simulator
            simulator = MarketSimulator(
                train_df, seed=42, start_market_price=start_price
            )
            
            # Fit on training data
            simulator.fit(train_df)
            print(f"  ✓ Model fitted on training data")
            
        except Exception as e:
            print(f"  ✗ Error during initialization/fitting: {str(e)}")
            continue
        
        # STEP C: Forecast
        try:
            print(f"  Running {ITERATIONS_PER_WINDOW} Monte Carlo iterations for 12 months...")
            price_paths = simulator.forecast_price(
                iterations=ITERATIONS_PER_WINDOW, steps=TEST_WINDOW
            )
            
            # Extract median path
            median_path = price_paths.median(axis=1).values
            print(f"  ✓ Forecast complete")
            print(f"    Median price range: ${median_path.min():,.2f} to ${median_path.max():,.2f}")
            
        except Exception as e:
            print(f"  ✗ Error during forecasting: {str(e)}")
            continue
        
        # STEP D: Extract Actual Prices & Store Results
        try:
            # Reconstruct actual prices from test slice
            actual_prices = extract_actual_prices(test_df)
            
            # If actual prices failed to compute, use lag_1 approach
            if actual_prices[0] == 0 or np.all(actual_prices == 0):
                actual_prices_list = [start_price]
                for ret in test_df['Log_Return_MoM'].fillna(0).values:
                    actual_prices_list.append(actual_prices_list[-1] * np.exp(ret))
                actual_prices = np.array(actual_prices_list[1:])  # Skip first anchor
            
            print(f"  Actual price range: ${actual_prices.min():,.2f} to ${actual_prices.max():,.2f}")
            
            # Store window results
            window_result = {
                'window_num': window_num,
                'start_date': test_df['date'].iloc[0],
                'end_date': test_df['date'].iloc[-1],
                'actual_prices': actual_prices,
                'predicted_prices': median_path,
                'dates': test_df['date'].values,
                'start_price': start_price
            }
            all_windows.append(window_result)
            
            print(f"  ✓ Window {window_num} results stored")
            
        except Exception as e:
            print(f"  ✗ Error during result processing: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        # Move to next window
        current_idx += TEST_WINDOW
    
    print(f"\n{'=' * 80}")
    print(f"✓ Completed {len(all_windows)} rolling windows")
    print(f"{'=' * 80}")
    
    return all_windows, df


def calculate_aggregate_metrics(all_windows):
    """Calculate error metrics on concatenated windows."""
    print("\n" + "=" * 80)
    print("STEP 3: Aggregate Metrics Calculation")
    print("=" * 80)
    
    # Concatenate all windows
    all_actual = np.concatenate([w['actual_prices'] for w in all_windows])
    all_predicted = np.concatenate([w['predicted_prices'] for w in all_windows])
    
    print(f"\n✓ Total predictions: {len(all_actual)} months")
    print(f"  Actual price range (full): ${all_actual.min():,.2f} to ${all_actual.max():,.2f}")
    print(f"  Predicted price range (full): ${all_predicted.min():,.2f} to ${all_predicted.max():,.2f}")
    
    # Ensure same length
    min_len = min(len(all_actual), len(all_predicted))
    all_actual = all_actual[:min_len]
    all_predicted = all_predicted[:min_len]
    
    errors = all_actual - all_predicted
    
    # 1. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(errors))
    
    # 2. MAPE (Mean Absolute Percentage Error)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((all_actual - all_predicted) / all_actual) * 100
        mape_values = mape_values[np.isfinite(mape_values)]
        mape = np.mean(mape_values) if len(mape_values) > 0 else np.inf
    
    # 3. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # 4. Directional Accuracy (Month-over-month)
    actual_direction = np.diff(all_actual)
    predicted_direction = np.diff(all_predicted)
    
    tolerance = 0.001
    correct_direction = (
        ((actual_direction > tolerance) & (predicted_direction > tolerance)) |
        ((actual_direction < -tolerance) & (predicted_direction < -tolerance)) |
        ((np.abs(actual_direction) <= tolerance) & (np.abs(predicted_direction) <= tolerance))
    )
    
    directional_accuracy = (np.sum(correct_direction) / len(correct_direction) * 100 
                           if len(correct_direction) > 0 else 0)
    
    # 5. Per-window metrics
    window_metrics = []
    for w in all_windows:
        w_errors = w['actual_prices'] - w['predicted_prices']
        w_mae = np.mean(np.abs(w_errors))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            w_mape_vals = np.abs((w['actual_prices'] - w['predicted_prices']) / w['actual_prices']) * 100
            w_mape_vals = w_mape_vals[np.isfinite(w_mape_vals)]
            w_mape = np.mean(w_mape_vals) if len(w_mape_vals) > 0 else np.nan
        
        w_rmse = np.sqrt(np.mean(w_errors ** 2))
        
        window_metrics.append({
            'window': w['window_num'],
            'MAE': w_mae,
            'MAPE': w_mape,
            'RMSE': w_rmse
        })
    
    window_metrics_df = pd.DataFrame(window_metrics)
    
    print("\n✓ Per-Window Performance:")
    print(window_metrics_df.to_string(index=False))
    
    metrics = {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'Directional_Accuracy': directional_accuracy,
        'window_metrics_df': window_metrics_df
    }
    
    return metrics


def plot_rolling_results(all_windows, metrics):
    """Create comprehensive visualization of rolling validation."""
    print("\n" + "=" * 80)
    print("STEP 4: Visualization")
    print("=" * 80)
    
    # Prepare data for plotting
    all_dates = np.concatenate([w['dates'] for w in all_windows])
    all_actual = np.concatenate([w['actual_prices'] for w in all_windows])
    all_predicted = np.concatenate([w['predicted_prices'] for w in all_windows])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # ===== SUBPLOT 1: Full Timeline with Rolling Windows =====
    ax = axes[0]
    
    # Plot actual prices (continuous line)
    ax.plot(all_dates, all_actual, 
            label='Actual Price', color='#2E7D32', linewidth=2.5, 
            marker='o', markersize=2, zorder=3)
    
    # Plot predicted prices per window (different colors)
    colors_window = plt.cm.tab10(np.linspace(0, 1, len(all_windows)))
    for i, w in enumerate(all_windows):
        # Get the indices for this window in the concatenated arrays
        window_start = i * TEST_WINDOW
        window_end = window_start + TEST_WINDOW
        
        ax.plot(w['dates'], w['predicted_prices'], 
                label=f"Window {w['window_num']}", 
                color=colors_window[i], linewidth=2, 
                marker='s', markersize=2, alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Market Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Rolling Walk-Forward Validation: Actual vs. Predicted Prices (12-Month Windows)',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='best', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e6:.2f}M' if x >= 1e6 else f'${x/1e3:.0f}K'
    ))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ===== SUBPLOT 2: Error Analysis =====
    ax = axes[1]
    
    errors = all_actual - all_predicted
    
    ax.bar(all_dates, errors, width=15, color=['#d32f2f' if e < 0 else '#1976d2' for e in errors],
           alpha=0.7, label='Prediction Error (Actual - Predicted)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(y=np.mean(errors), color='green', linestyle='--', linewidth=1.5, label=f'Mean Error: ${np.mean(errors):,.0f}')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error ($)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Errors by Month (Red=Overpredicted, Blue=Underpredicted)',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'${x/1e3:.0f}K'
    ))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{RESULTS_DIR}/rolling_validation_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")
    plt.close()
    
    # Create a second figure for metrics summary
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    metrics_text = f"""
    ROLLING WALK-FORWARD VALIDATION RESULTS
    {'═' * 50}
    
    Configuration:
      • Test Window:           {TEST_WINDOW} months
      • Total Test Months:     {TOTAL_TEST_MONTHS} months
      • Number of Windows:     {len(all_windows)}
      • Iterations per Window: {ITERATIONS_PER_WINDOW} Monte Carlo simulations
    
    Aggregate Metrics:
      • MAE (Mean Absolute Error):        ${metrics['MAE']:>15,.2f}
      • MAPE (Mean Absolute % Error):    {metrics['MAPE']:>15.2f}%
      • RMSE (Root Mean Squared Error):   ${metrics['RMSE']:>15,.2f}
      • Directional Accuracy:            {metrics['Directional_Accuracy']:>15.2f}%
    
    Statistical Summary:
      • Mean Prediction Error:           ${np.mean(all_actual - all_predicted):>15,.2f}
      • Std Dev of Errors:               ${np.std(all_actual - all_predicted):>15,.2f}
      • Max Overpredict:                 ${np.min(all_actual - all_predicted):>15,.2f}
      • Max Underpredict:                ${np.max(all_actual - all_predicted):>15,.2f}
    
    Price Movement:
      • Actual Price Range:              ${all_actual.min():>15,.2f} to ${all_actual.max():,.2f}
      • Predicted Price Range:           ${all_predicted.min():>15,.2f} to ${all_predicted.max():,.2f}
      • Actual Total Return (%):         {((all_actual[-1] / all_actual[0]) - 1) * 100:>14.2f}%
      • Predicted Total Return (%):      {((all_predicted[-1] / all_predicted[0]) - 1) * 100:>14.2f}%
    
    Interpretation:
      ✓ Lower MAE/RMSE = More accurate absolute predictions
      ✓ Lower MAPE = Better percentage accuracy
      ✓ High Directional Accuracy = Model captures market direction
      ✓ Running predictions address "Anchor Drift" from long forecasts
    """
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    metrics_output = f"{RESULTS_DIR}/rolling_validation_metrics.png"
    plt.savefig(metrics_output, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics summary saved to: {metrics_output}")
    plt.close()


def print_final_summary(all_windows, metrics):
    """Print comprehensive final summary."""
    print("\n" + "=" * 80)
    print("FINAL RESULTS - ROLLING WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    
    all_actual = np.concatenate([w['actual_prices'] for w in all_windows])
    all_predicted = np.concatenate([w['predicted_prices'] for w in all_windows])
    
    print(f"\nTest Configuration:")
    print(f"  • Test Window Size:       {TEST_WINDOW} months")
    print(f"  • Total Test Period:      {TOTAL_TEST_MONTHS} months")
    print(f"  • Number of Windows:      {len(all_windows)}")
    print(f"  • Iterations per Window:  {ITERATIONS_PER_WINDOW} Monte Carlo paths")
    
    print(f"\nPerformance Metrics (Aggregate):")
    print(f"  • MAE:                    ${metrics['MAE']:>15,.2f}")
    print(f"  • MAPE:                   {metrics['MAPE']:>15.2f}%")
    print(f"  • RMSE:                   ${metrics['RMSE']:>15,.2f}")
    print(f"  • Directional Accuracy:   {metrics['Directional_Accuracy']:>15.2f}%")
    
    print(f"\nPrice Movement Analysis:")
    print(f"  • Actual ending price:    ${all_actual[-1]:>15,.2f}")
    print(f"  • Predicted ending price: ${all_predicted[-1]:>15,.2f}")
    print(f"  • Actual return:          {((all_actual[-1] / all_actual[0]) - 1) * 100:>15.2f}%")
    print(f"  • Predicted return:       {((all_predicted[-1] / all_predicted[0]) - 1) * 100:>15.2f}%")
    
    print(f"\nPer-Window Summary:")
    window_df = metrics['window_metrics_df']
    print(window_df.to_string(index=False))
    
    print(f"\n" + "=" * 80)
    print("✓ Rolling Walk-Forward Validation Complete")
    print("=" * 80 + "\n")


def main():
    """Main execution flow."""
    try:
        # Run rolling validation
        all_windows, df = run_rolling_validation()
        
        if len(all_windows) == 0:
            print("\n❌ No windows were successfully processed. Exiting.")
            return 1
        
        # Calculate metrics
        metrics = calculate_aggregate_metrics(all_windows)
        
        # Plot results
        plot_rolling_results(all_windows, metrics)
        
        # Print summary
        print_final_summary(all_windows, metrics)
        
        # Save detailed results to CSV
        all_actual = np.concatenate([w['actual_prices'] for w in all_windows])
        all_predicted = np.concatenate([w['predicted_prices'] for w in all_windows])
        all_dates = np.concatenate([w['dates'] for w in all_windows])
        
        results_df = pd.DataFrame({
            'Date': all_dates,
            'Actual_Price': all_actual,
            'Predicted_Price': all_predicted,
            'Error': all_actual - all_predicted,
            'Error_Percent': ((all_actual - all_predicted) / all_actual * 100)
        })
        
        results_csv = f"{RESULTS_DIR}/rolling_validation_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"✓ Detailed CSV results saved to: {results_csv}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
