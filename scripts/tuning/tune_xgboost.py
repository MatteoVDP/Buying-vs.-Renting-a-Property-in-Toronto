import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import time
from market_simulator import MarketSimulator

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS
# ============================================================================

# TEST MODE: Set to True for quick debugging with reduced grid and single window
TEST_MODE = True

# Parameter grid for hyperparameter tuning
if TEST_MODE:
    # Reduced grid: 4 representative combinations
    PARAM_GRID = {
        'n_estimators': [100, 300],
        'learning_rate': [0.001, 0.1]
    }
else:
    # Full grid: 5 x 4 = 20 combinations
    # (kept here for reference - do not delete)
    # PARAM_GRID = {
    #     'n_estimators': [100, 200, 300, 400, 500],
    #     'learning_rate': [0.0001, 0.001, 0.01, 0.1]
    # }
    PARAM_GRID = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1]
    }

# --- TARGETED RUN CONFIGURATION ------------------------------------------------
# When USE_TARGET_COMBINATIONS=True the script will run only the explicit combos
USE_TARGET_COMBINATIONS = True
TARGET_COMBINATIONS = [
    (200, 0.05),   # The Baseline Control
    (300, 0.05),   # The Efficient Midpoint
    (400, 0.025),  # The Goldilocks Zone
    (500, 0.05),   # The COVID-Catcher
    (500, 0.01)    # The Slow & Cautious Learner
]

# Rolling Walk-Forward Configuration
TEST_WINDOW = 12  # 12-month test window
if TEST_MODE:
    TOTAL_TEST_MONTHS = 12  # Single window for testing
else:
    TOTAL_TEST_MONTHS = 60  # Total months to test over (5 years, includes COVID-era volatility)
SEED = 42

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    if (~mask).any():
        # For zero values, use MAE instead of MAPE
        mape_values = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    else:
        mape_values = np.abs((y_true - y_pred) / y_true)
    
    return np.mean(mape_values) * 100


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rolling_walk_forward_validation(df, n_estimators, learning_rate):
    """
    Perform rolling walk-forward validation with 12-month test windows.
    Converts log returns to actual dollar prices for accurate error calculation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data sorted by date, with 'date' column
    n_estimators : int
        XGBoost n_estimators hyperparameter
    learning_rate : float
        XGBoost learning_rate hyperparameter
    
    Returns:
    --------
    mape : float
        Aggregate MAPE across all test windows (in dollar prices)
    rmse : float
        Aggregate RMSE across all test windows (in dollar prices)
    """
    
    # Ensure data is sorted by date
    df = df.copy().sort_values('date').reset_index(drop=True)
    n_total = len(df)
    
    # Calculate the start index for the first training window
    start_test_idx = n_total - TOTAL_TEST_MONTHS
    
    if start_test_idx < 0:
        raise ValueError(f"Not enough data for walk-forward validation. Need at least {TOTAL_TEST_MONTHS} test months.")
    
    all_predicted_prices = []
    all_actual_prices = []
    
    # Rolling walk-forward: slide TEST_WINDOW (12 months) across the test period
    # Step by TEST_WINDOW months to create non-overlapping 12-month windows.
    for test_start_idx in range(start_test_idx, n_total - TEST_WINDOW + 1, TEST_WINDOW):
        test_end_idx = test_start_idx + TEST_WINDOW
        
        # Training data: everything before test window
        train_df = df.iloc[:test_start_idx].copy()
        test_df = df.iloc[test_start_idx:test_end_idx].copy()
        
        # Skip if not enough training data
        if len(train_df) < 50:
            continue
        
        try:
            # Extract anchor price (starting price for this window)
            # Use the lag_1 column from the last training row
            price_col_lag1 = 'market_price_target_average,_detached_single_family_homes_lag_1'
            if price_col_lag1 in train_df.columns and pd.notna(train_df[price_col_lag1].iloc[-1]):
                anchor_price = float(train_df[price_col_lag1].iloc[-1])
            else:
                # Fallback: search for any price column
                price_cols = [c for c in train_df.columns if 'market_price' in c.lower() and 'lag' not in c.lower()]
                if price_cols:
                    anchor_price = float(train_df[price_cols[0]].dropna().iloc[-1])
                else:
                    anchor_price = 1090326.0  # Default starting price
            
            # Initialize MarketSimulator with hyperparameters
            simulator = MarketSimulator(
                train_df,
                seed=SEED,
                start_market_price=anchor_price,
                xgb_n_estimators=n_estimators,
                xgb_learning_rate=learning_rate
            )
            
            # Fit the model on training data
            simulator.fit(train_df)
            
            # Forecast for the test window
            forecast_result = simulator.forecast_price(
                steps=TEST_WINDOW,
                iterations=1
            )
            
            # Check forecast validity
            if forecast_result is None or not isinstance(forecast_result, pd.DataFrame):
                continue
            
            if 'iter_0' not in forecast_result.columns:
                continue
            
            # Extract predicted prices (already in dollars from forecast_price)
            predicted_prices = forecast_result['iter_0'].values[:TEST_WINDOW]
            
            # Reconstruct actual prices from log returns
            actual_prices = []
            current_price = anchor_price
            for log_return in test_df['Log_Return_MoM'].fillna(0).values:
                current_price = current_price * np.exp(log_return)
                actual_prices.append(current_price)
            actual_prices = np.array(actual_prices)
            
            # Validate lengths match
            if len(predicted_prices) != len(actual_prices):
                continue
            
            all_predicted_prices.extend(predicted_prices)
            all_actual_prices.extend(actual_prices)
        
        except Exception as e:
            # Skip this window if there's an error
            continue
    
    if not all_predicted_prices:
        return np.nan, np.nan
    
    all_predicted_prices = np.array(all_predicted_prices)
    all_actual_prices = np.array(all_actual_prices)
    
    # Calculate metrics on dollar prices (not log returns)
    mape = calculate_mape(all_actual_prices, all_predicted_prices)
    rmse = calculate_rmse(all_actual_prices, all_predicted_prices)
    
    return mape, rmse

# ============================================================================
# COMBO EVALUATION FUNCTION FOR MULTIPROCESSING
# ============================================================================

def evaluate_combo(n_est, lr, df):
    """
    Evaluate a single hyperparameter combination.
    
    Parameters:
    -----------
    n_est : int
        Number of estimators for XGBoost
    lr : float
        Learning rate for XGBoost
    df : pd.DataFrame
        Data for rolling walk-forward validation
    
    Returns:
    --------
    dict : Result dictionary with n_estimators, learning_rate, MAPE, RMSE
    """
    mape, rmse = rolling_walk_forward_validation(df, n_est, lr)
    return {
        'n_estimators': n_est,
        'learning_rate': lr,
        'MAPE': mape,
        'RMSE': rmse
    }

# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def main():
    # Load data
    print("Loading processed data...")
    df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
    
    if 'date' not in df.columns:
        raise ValueError("'date' column not found in processed_data.csv")
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
    
    # Results storage
    results = []

    # Build combinations list depending on mode
    if USE_TARGET_COMBINATIONS:
        combinations = TARGET_COMBINATIONS
        total_combinations = len(combinations)
        # For targeted runs, use 60 months (5 windows)
        global TOTAL_TEST_MONTHS
        TOTAL_TEST_MONTHS = 60
        mode = f"TARGET COMBINATIONS ({total_combinations} combos, 60 months)"
    else:
        combinations = [
            (n_est, lr)
            for n_est in PARAM_GRID['n_estimators']
            for lr in PARAM_GRID['learning_rate']
        ]
        total_combinations = len(combinations)
        mode = "TEST MODE (4 combos, 1 window)" if TEST_MODE else "FULL MODE (20 combos, rolling windows)"

    completed_count = 0
    if USE_TARGET_COMBINATIONS:
        print(f"Target combinations ({total_combinations}): {combinations}")
    else:
        print(f"Parameter grid: {len(PARAM_GRID['n_estimators'])} x {len(PARAM_GRID['learning_rate'])} = {total_combinations} combinations ({mode})")
    print(f"Starting grid search with {total_combinations} combinations...\n")

    # Sequential Grid Search
    start_time = time.time()

    for idx, (n_est, lr) in enumerate(combinations, start=1):
        completed_count = idx
        combo_start = time.time()

        # Notify START of this combination
        print(f"\n{'='*70}")
        print(f"[{completed_count}/{total_combinations}] STARTING: lr={lr:.4f}, trees={n_est}")
        print(f"{'='*70}", flush=True)

        try:
            result = evaluate_combo(n_est, lr, df)
            mape = result['MAPE']
            rmse = result['RMSE']
            combo_time = time.time() - combo_start

            # Notify COMPLETION of this combination
            print(f"[{completed_count}/{total_combinations}] COMPLETED: lr={lr:.4f}, trees={n_est}")
            print(f"  ├─ MAPE: {mape:.2f}%")
            print(f"  ├─ RMSE: {rmse:.4f}")
            print(f"  └─ Time: {combo_time:.1f}s")
            print(f"{'='*70}\n", flush=True)

            results.append(result)

        except Exception as e:
            combo_time = time.time() - combo_start
            print(f"[{completed_count}/{total_combinations}] ERROR: lr={lr:.4f}, trees={n_est}")
            print(f"  └─ {str(e)[:100]}")
            print(f"{'='*70}\n", flush=True)
            # Add NaN result to maintain structure
            results.append({
                'n_estimators': n_est,
                'learning_rate': lr,
                'MAPE': np.nan,
                'RMSE': np.nan
            })
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/results/tuning_results.csv', index=False)
    print(f"\nResults saved to results/tuning_results.csv")
    
    # Find best parameters (handling NaN values)
    valid_results = results_df[results_df['MAPE'].notna()]
    
    if len(valid_results) == 0:
        print("\nWARNING: All hyperparameter combinations failed. Check error logs above.")
        print("="*70)
    else:
        best_idx = valid_results['MAPE'].idxmin()
        best_row = results_df.loc[best_idx]
        
        print("\n" + "="*70)
        print("BEST PARAMETERS (by lowest MAPE)")
        print("="*70)
        print(f"n_estimators: {int(best_row['n_estimators'])}")
        print(f"learning_rate: {best_row['learning_rate']:.4f}")
        print(f"MAPE: {best_row['MAPE']:.2f}%")
        print(f"RMSE: {best_row['RMSE']:.4f}")
        print("="*70)
    
    # Generate Heatmap
    print(f"\nGenerating heatmap...")
    
    # Pivot for heatmap
    heatmap_data = results_df.pivot_table(
        index='learning_rate',
        columns='n_estimators',
        values='MAPE'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap with annotations
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Red (high) to Green (low)
        cbar_kws={'label': 'MAPE (%)'},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title('XGBoost Hyperparameter Tuning: MAPE Heatmap (Rolling Walk-Forward)', fontsize=14, fontweight='bold')
    ax.set_xlabel('n_estimators', fontsize=12, fontweight='bold')
    ax.set_ylabel('learning_rate', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/results/tuning_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to results/tuning_heatmap.png")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    valid_results_sorted = valid_results.sort_values('MAPE').head(10)
    if len(valid_results_sorted) > 0:
        print(valid_results_sorted.to_string(index=False))
    else:
        print("No valid results to display.")
    print("="*70)


if __name__ == '__main__':
    main()
