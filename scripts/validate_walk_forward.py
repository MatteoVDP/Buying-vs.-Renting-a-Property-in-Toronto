"""
Walk-Forward Validation Script for Long-Term Real Estate Forecasting
=====================================================================
Executes Time Series Cross-Validation with Rolling Origin across 6 staggered
20-year periods, testing the MarketSimulator's autoregressive forecasting capability.

Methodology:
- Train on all historical data up to a specific cutoff date
- Blindly forecast 240 months (20 years) into the future
- Compare median of 10 Monte Carlo paths against actual outcomes
- Measure MAPE, RMSE, Directional Accuracy, CAGR Error, and Distribution Coverage
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
from dateutil.relativedelta import relativedelta
import inspect
import warnings
try:
    from market_simulator import MarketSimulator
except ModuleNotFoundError:
    from scripts.market_simulator import MarketSimulator

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_SEED = 42  # Global seed for reproducibility
FORECAST_HORIZON = 240  # 20 years = 240 months
MONTE_CARLO_ITERATIONS = 3  # Number of simulation paths per fold
RESULTS_DIR = "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/results"
DATA_PATH = "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv"
AGGREGATE_FACT_SHEET_OUTPUT = f"{RESULTS_DIR}/walk_forward_aggregate_fact_sheet.png"

# Define the 6 validation periods
VALIDATION_FOLDS = [
    #{"train_end": "1985-03-01", "test_start": "1985-04-01", "test_end": "2005-03-01"},
    #{"train_end": "1989-03-01", "test_start": "1989-04-01", "test_end": "2009-03-01"},
    {"train_end": "1993-03-01", "test_start": "1993-04-01", "test_end": "2013-03-01"},
    {"train_end": "1997-03-01", "test_start": "1997-04-01", "test_end": "2017-03-01"},
    {"train_end": "2001-03-01", "test_start": "2001-04-01", "test_end": "2021-03-01"},
    {"train_end": "2005-03-01", "test_start": "2005-04-01", "test_end": "2025-03-01"},
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data():
    """Load and prepare the processed dataset."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = add_reconstructed_market_prices(df)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print()
    
    return df


def get_simulator_reference_price():
    """Read MarketSimulator's default starting market price from its signature."""
    return float(inspect.signature(MarketSimulator.__init__).parameters['start_market_price'].default)


def add_reconstructed_market_prices(df):
    """
    Reconstruct absolute market prices from Log_Return_MoM using the simulator's
    default reference price anchored at the last historical observation.
    """
    if 'Log_Return_MoM' not in df.columns:
        raise ValueError("Dataset must include 'Log_Return_MoM' for price reconstruction")

    reference_price = get_simulator_reference_price()
    returns = df['Log_Return_MoM'].fillna(0).values.astype(float)

    reconstructed_prices = np.zeros(len(df), dtype=float)
    reconstructed_prices[-1] = reference_price

    for idx in range(len(df) - 1, 0, -1):
        reconstructed_prices[idx - 1] = reconstructed_prices[idx] / np.exp(returns[idx])

    df = df.copy()
    df['Reconstructed_Market_Price'] = reconstructed_prices
    return df


def split_data(df, train_end_date):
    """
    Split dataset at the specified cutoff date.
    
    Args:
        df: Full dataset
        train_end_date: Last date to include in training (e.g., '1986-03-01')
    
    Returns:
        train_df: Training data up to and including train_end_date
        test_df: Testing data after train_end_date
        anchor_price: The market price at the last training month
    """
    cutoff = pd.to_datetime(train_end_date)
    
    train_df = df[df['date'] <= cutoff].copy()
    test_df = df[df['date'] > cutoff].copy()
    
    # Compute anchor price from reconstructed absolute market price
    anchor_price = float(train_df['Reconstructed_Market_Price'].iloc[-1])
    
    return train_df, test_df, anchor_price


def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_directional_accuracy(actual, predicted):
    """
    Calculate the percentage of months where the forecasted direction 
    (up/down relative to previous month) matches the actual direction.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate month-over-month changes
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    
    # Compare directions
    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual_direction)
    
    return (correct / total) * 100 if total > 0 else 0.0


def calculate_cagr(start_price, end_price, years):
    """Calculate Compound Annual Growth Rate."""
    if start_price <= 0 or end_price <= 0:
        return 0.0
    return (np.power(end_price / start_price, 1 / years) - 1) * 100


def calculate_distribution_coverage(actual, all_paths):
    """
    Calculate the percentage of actual prices that fall within the 
    min-max envelope of the Monte Carlo simulations.
    
    Args:
        actual: Array of actual prices
        all_paths: DataFrame with columns as iterations (iter_0, iter_1, ...)
    
    Returns:
        coverage_pct: Percentage of actual values within the simulated range
    """
    actual = np.array(actual)
    
    # Get min and max bounds from all iterations
    path_min = all_paths.min(axis=1).values
    path_max = all_paths.max(axis=1).values
    
    # Check how many actual prices fall within bounds
    within_bounds = (actual >= path_min) & (actual <= path_max)
    coverage_pct = (np.sum(within_bounds) / len(actual)) * 100
    
    return coverage_pct


def reconstruct_actual_prices(test_df):
    """
    Reconstruct actual market prices from log returns.
    
    Args:
        test_df: Test portion of the dataset
    
    Returns:
        prices: Array of reconstructed prices
    """
    return test_df['Reconstructed_Market_Price'].values.astype(float)


def extract_price_paths(forecast_result):
    """
    Normalize MarketSimulator.forecast_price output across versions.

    New format: dict with {'price_paths': DataFrame, 'full_history': DataFrame}
    Legacy format: DataFrame of price paths
    """
    if isinstance(forecast_result, dict):
        if 'price_paths' not in forecast_result:
            raise ValueError("forecast_price() dict output missing 'price_paths'")
        return forecast_result['price_paths']
    return forecast_result


def run_fold_validation(fold_idx, fold_config, df):
    """
    Execute validation for a single fold.
    
    Args:
        fold_idx: Index of the fold (0-5)
        fold_config: Dictionary with train_end, test_start, test_end dates
        df: Full dataset
    
    Returns:
        results: Dictionary containing all metrics and data for this fold
    """
    print("=" * 80)
    print(f"FOLD {fold_idx + 1}/6: Train up to {fold_config['train_end']}")
    print(f"          Forecast: {fold_config['test_start']} to {fold_config['test_end']}")
    print("=" * 80)

    # Step 1: Split data
    train_df, test_df, anchor_price = split_data(df, fold_config['train_end'])
    anchor_date = train_df['date'].iloc[-1]

    print(f"✓ Training samples: {len(train_df)} months")
    print(f"✓ Testing samples: {len(test_df)} months (target: {FORECAST_HORIZON})")
    print(f"✓ Anchor date: {anchor_date.date()}")
    print(f"✓ Anchor price: ${anchor_price:,.2f}")

    # Verify we have enough test data
    if len(test_df) < FORECAST_HORIZON:
        print(f"⚠ WARNING: Partial test period ({len(test_df)} < {FORECAST_HORIZON} months)")
        print(f"   Will evaluate on available {len(test_df)} months")
        actual_horizon = len(test_df)
    else:
        actual_horizon = FORECAST_HORIZON

    # Truncate test data to the actual horizon
    test_df = test_df.iloc[:actual_horizon].copy()

    # Step 2: Initialize and train the MarketSimulator (ONCE per fold)
    print("\n[1/4] Initializing MarketSimulator...")
    simulator = MarketSimulator(
        df=train_df,
        seed=BASE_SEED,
        start_market_price=anchor_price
    )

    print("[2/4] Training model on historical data...")
    simulator.fit()

    # Step 3: Run Monte Carlo simulations (10 paths in one call)
    print(f"[3/4] Running {MONTE_CARLO_ITERATIONS} Monte Carlo simulations...")
    np.random.seed(BASE_SEED)
    try:
        forecast_result = simulator.forecast_price(
            iterations=MONTE_CARLO_ITERATIONS,
            steps=actual_horizon
        )
    except Exception as e:
        raise RuntimeError(
            f"Fold {fold_idx + 1} failed during MarketSimulator.forecast_price() "
            f"for period {fold_config['test_start']} to {fold_config['test_end']}"
        ) from e

    all_forecast_paths = extract_price_paths(forecast_result)

    # Step 4: Calculate median forecast
    median_forecast = all_forecast_paths.median(axis=1).values

    # Step 5: Reconstruct actual prices from test data
    print("[4/4] Calculating performance metrics...")
    actual_prices = reconstruct_actual_prices(test_df)

    # Step 6: Calculate all metrics
    mape = calculate_mape(actual_prices, median_forecast)
    rmse = calculate_rmse(actual_prices, median_forecast)
    directional_acc = calculate_directional_accuracy(actual_prices, median_forecast)

    # CAGR calculations
    years = actual_horizon / 12.0  # Actual years covered
    actual_cagr = calculate_cagr(actual_prices[0], actual_prices[-1], years)
    forecast_cagr = calculate_cagr(median_forecast[0], median_forecast[-1], years)
    cagr_error = forecast_cagr - actual_cagr

    # Distribution coverage
    coverage = calculate_distribution_coverage(actual_prices, all_forecast_paths)

    print("✓ Fold performance metrics computed (used in aggregate fact sheet)")
    print()

    # Return all data for aggregation and visualization
    return {
        'fold_idx': fold_idx,
        'train_end': fold_config['train_end'],
        'test_start': fold_config['test_start'],
        'test_end': fold_config['test_end'],
        'actual_horizon': actual_horizon,
        'actual_prices': actual_prices,
        'median_forecast': median_forecast,
        'all_paths': all_forecast_paths,
        'metrics': {
            'mape': mape,
            'rmse': rmse,
            'directional_accuracy': directional_acc,
            'actual_cagr': actual_cagr,
            'forecast_cagr': forecast_cagr,
            'cagr_error': cagr_error,
            'coverage': coverage
        }
    }


def _draw_fact_card(ax, x, y, w, h, title, value, subtitle):
    """Draw a compact fact card inside a fact-sheet canvas."""
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#1f4e79',
        linewidth=1.8,
    )
    ax.add_patch(card)
    ax.text(x + 0.02, y + h - 0.04, title, transform=ax.transAxes, fontsize=10, fontweight='bold', color='#334e68')
    ax.text(x + 0.02, y + h * 0.50, value, transform=ax.transAxes, fontsize=17, fontweight='bold', color='#102a43')
    ax.text(x + 0.02, y + 0.03, subtitle, transform=ax.transAxes, fontsize=9.2, color='#627d98')


def export_aggregate_fact_sheet(all_results, output_path):
    """Export aggregate walk-forward statistics as a visual fact sheet."""
    if not all_results:
        raise ValueError('Cannot build aggregate fact sheet from empty results')

    mapes = np.array([r['metrics']['mape'] for r in all_results], dtype=float)
    rmses = np.array([r['metrics']['rmse'] for r in all_results], dtype=float)
    directional_accs = np.array([r['metrics']['directional_accuracy'] for r in all_results], dtype=float)
    cagr_errors = np.array([r['metrics']['cagr_error'] for r in all_results], dtype=float)
    coverages = np.array([r['metrics']['coverage'] for r in all_results], dtype=float)
    horizons = np.array([r['actual_horizon'] for r in all_results], dtype=int)

    best_mape = min(all_results, key=lambda r: r['metrics']['mape'])
    worst_mape = max(all_results, key=lambda r: r['metrics']['mape'])
    best_coverage = max(all_results, key=lambda r: r['metrics']['coverage'])
    worst_coverage = min(all_results, key=lambda r: r['metrics']['coverage'])

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#eef3f9')
    ax.set_facecolor('#eef3f9')
    ax.axis('off')

    ax.text(
        0.02,
        0.965,
        'Walk-Forward Aggregate Performance Fact Sheet',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        color='#102a43',
        va='top',
    )
    ax.text(
        0.02,
        0.925,
        (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"Valid folds: {len(all_results)}/{len(VALIDATION_FOLDS)}  |  "
            f"Average horizon: {np.mean(horizons):.1f} months"
        ),
        transform=ax.transAxes,
        fontsize=10,
        color='#486581',
    )

    card_w = 0.305
    card_h = 0.165
    x_positions = [0.02, 0.347, 0.674]
    y_top = 0.70
    y_bottom = 0.50

    cards = [
        ('Average MAPE', f"{np.mean(mapes):.2f}%", f"Range: {np.min(mapes):.2f}% to {np.max(mapes):.2f}%"),
        ('Median MAPE', f"{np.median(mapes):.2f}%", f"Std Dev: {np.std(mapes, ddof=0):.2f} pts"),
        ('Average RMSE', f"${np.mean(rmses):,.0f}", f"Range: ${np.min(rmses):,.0f} to ${np.max(rmses):,.0f}"),
        (
            'Average Directional Accuracy',
            f"{np.mean(directional_accs):.2f}%",
            f"Range: {np.min(directional_accs):.2f}% to {np.max(directional_accs):.2f}%",
        ),
        ('Average Coverage', f"{np.mean(coverages):.2f}%", f"Range: {np.min(coverages):.2f}% to {np.max(coverages):.2f}%"),
        ('Average CAGR Error', f"{np.mean(cagr_errors):+.2f}%", f"Range: {np.min(cagr_errors):+.2f}% to {np.max(cagr_errors):+.2f}%"),
    ]

    for idx, (title, value, subtitle) in enumerate(cards[:3]):
        _draw_fact_card(ax, x_positions[idx], y_top, card_w, card_h, title, value, subtitle)
    for idx, (title, value, subtitle) in enumerate(cards[3:]):
        _draw_fact_card(ax, x_positions[idx], y_bottom, card_w, card_h, title, value, subtitle)

    left_panel = FancyBboxPatch(
        (0.02, 0.12),
        0.47,
        0.30,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#829ab1',
        linewidth=1.4,
    )
    right_panel = FancyBboxPatch(
        (0.51, 0.12),
        0.47,
        0.30,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='#829ab1',
        linewidth=1.4,
    )
    ax.add_patch(left_panel)
    ax.add_patch(right_panel)

    ax.text(0.04, 0.395, 'Range Snapshot', transform=ax.transAxes, fontsize=12, fontweight='bold', color='#102a43')
    ax.text(
        0.04,
        0.36,
        '\n'.join(
            [
                f"• MAPE spread: {np.max(mapes) - np.min(mapes):.2f} percentage points",
                f"• Directional accuracy spread: {np.max(directional_accs) - np.min(directional_accs):.2f} points",
                f"• Coverage spread: {np.max(coverages) - np.min(coverages):.2f} points",
                f"• RMSE spread: ${np.max(rmses) - np.min(rmses):,.0f}",
                f"• CAGR error spread: {np.max(cagr_errors) - np.min(cagr_errors):.2f} points",
            ]
        ),
        transform=ax.transAxes,
        fontsize=10,
        color='#334e68',
        va='top',
    )

    ax.text(0.53, 0.395, 'Fold Highlights', transform=ax.transAxes, fontsize=12, fontweight='bold', color='#102a43')
    ax.text(
        0.53,
        0.36,
        '\n'.join(
            [
                (
                    f"• Best MAPE: Fold {best_mape['fold_idx'] + 1} "
                    f"({best_mape['test_start'][:7]} to {best_mape['test_end'][:7]}) "
                    f"at {best_mape['metrics']['mape']:.2f}%"
                ),
                (
                    f"• Worst MAPE: Fold {worst_mape['fold_idx'] + 1} "
                    f"({worst_mape['test_start'][:7]} to {worst_mape['test_end'][:7]}) "
                    f"at {worst_mape['metrics']['mape']:.2f}%"
                ),
                (
                    f"• Best coverage: Fold {best_coverage['fold_idx'] + 1} "
                    f"at {best_coverage['metrics']['coverage']:.2f}%"
                ),
                (
                    f"• Lowest coverage: Fold {worst_coverage['fold_idx'] + 1} "
                    f"at {worst_coverage['metrics']['coverage']:.2f}%"
                ),
                (
                    f"• Current average directional accuracy suggests "
                    f"{np.mean(directional_accs):.2f}% month-to-month direction match"
                ),
            ]
        ),
        transform=ax.transAxes,
        fontsize=10,
        color='#334e68',
        va='top',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Aggregate fact sheet saved to: {output_path}")
    print()


def aggregate_results(all_results):
    """Calculate overall statistics across all folds."""
    print("=" * 80)
    print("OVERALL AGGREGATE STATISTICS")
    print("=" * 80)
    
    valid_results = all_results
    
    # Extract metrics
    mapes = [r['metrics']['mape'] for r in valid_results]
    directional_accs = [r['metrics']['directional_accuracy'] for r in valid_results]
    cagr_errors = [r['metrics']['cagr_error'] for r in valid_results]
    coverages = [r['metrics']['coverage'] for r in valid_results]
    
    print(f"Valid Folds:                {len(valid_results)}/6")
    print(f"Overall Average MAPE:       {np.mean(mapes):.2f}%")
    print(f"Overall Average Dir. Acc:   {np.mean(directional_accs):.2f}%")
    print(f"Overall Average CAGR Error: {np.mean(cagr_errors):+.2f}%")
    print(f"Overall Average Coverage:   {np.mean(coverages):.2f}%")
    print("=" * 80)
    print()


def visualize_results(all_results):
    """Generate 3x2 grid visualization of all folds."""
    print("=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)
    
    valid_results = all_results
    
    # Create 3x2 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for idx, result in enumerate(valid_results):
        ax = axes[idx]
        
        # Extract data
        actual = result['actual_prices']
        median_fc = result['median_forecast']
        all_paths = result['all_paths']
        metrics = result['metrics']
        actual_horizon = result['actual_horizon']
        years = actual_horizon / 12.0
        
        # Generate month indices for x-axis (0 to 239)
        months = np.arange(len(actual))
        
        # Plot all Monte Carlo paths (light gray)
        for col in all_paths.columns:
            ax.plot(months, all_paths[col].values, color='gray', alpha=0.3, linewidth=0.8)
        
        # Plot median forecast (bold blue)
        ax.plot(months, median_fc, color='blue', linewidth=2.5, label='Median Forecast')
        
        # Plot actual prices (bold red)
        ax.plot(months, actual, color='red', linewidth=2.5, label='Actual Prices')
        
        # Formatting
        ax.set_title(
                        f"Period {idx + 1}: {result['test_start'][:7]} to {result['test_end'][:7]} ({actual_horizon} months)\n"
                        f"MAPE: {metrics['mape']:.2f}% | {years:.1f}-yr CAGR Error: {metrics['cagr_error']:+.2f}%",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel("Months into Forecast", fontsize=10)
        ax.set_ylabel("Market Price ($)", fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')
        
        # Format y-axis to show prices with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Hide any unused subplots
    for idx in range(len(valid_results), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{RESULTS_DIR}/walk_forward_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION: 20-YEAR AUTOREGRESSIVE FORECASTING")
    print("=" * 80)
    print(f"Base Seed:         {BASE_SEED}")
    print(f"Forecast Horizon:  {FORECAST_HORIZON} months (20 years)")
    print(f"MC Iterations:     {MONTE_CARLO_ITERATIONS}")
    print(f"Validation Folds:  {len(VALIDATION_FOLDS)}")
    print("=" * 80)
    print()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    df = load_data()
    
    # Run validation for each fold
    all_results = []
    for idx, fold_config in enumerate(VALIDATION_FOLDS):
        result = run_fold_validation(idx, fold_config, df)
        all_results.append(result)

    # Export aggregate metrics fact sheet
    export_aggregate_fact_sheet(all_results, AGGREGATE_FACT_SHEET_OUTPUT)
    
    # Aggregate statistics
    aggregate_results(all_results)
    
    # Generate visualization
    visualize_results(all_results)
    
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
