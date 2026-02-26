"""
Diagnostic script to understand why predictions explode to 300-1000% above reality.
"""
import numpy as np
import pandas as pd
import warnings
from market_simulator import MarketSimulator

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Use first validation fold: train up to 1985-03-01
train_df = df[df['date'] <= '1985-03-01'].copy()
test_df = df[(df['date'] > '1985-03-01') & (df['date'] <= '2005-03-01')].copy()

# Get anchor price
last_row = train_df.iloc[-1]
log_price_current = last_row['Log_Price_lag_1'] + last_row['Log_Return_MoM']
anchor_price = np.exp(log_price_current)

print("=" * 80)
print("DIAGNOSTIC: Why are predictions exploding?")
print("=" * 80)
print(f"Anchor date: {train_df['date'].iloc[-1].date()}")
print(f"Anchor price: ${anchor_price:,.2f}")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)} (20 years)")
print()

# Train model
print("Training model...")
simulator = MarketSimulator(
    df=train_df,
    seed=42,
    start_market_price=anchor_price
)
simulator.fit()
print()

# Run ONE simulation with detailed tracking
print("=" * 80)
print("Running ONE simulation with detailed diagnostics...")
print("=" * 80)

# Manually run forecast to capture intermediate values
np.random.seed(42)
sim_world = simulator.simulate_exogenous(steps=240)  # 20 years
current_hist = simulator.df.copy()

# Track the predictions
log_returns_xgb = []
log_returns_with_sentiment = []
sentiment_scores = []
prices = []

sentiment_score = 0.0

# Get initial log price
if 'Log_Price' in current_hist.columns and not current_hist['Log_Price'].dropna().empty:
    current_log_price = float(current_hist['Log_Price'].dropna().iloc[-1])
else:
    current_log_price = float(np.log(simulator.start_market_price))

print(f"Initial log price: {current_log_price:.6f}")
print(f"Initial price: ${np.exp(current_log_price):,.2f}")
print()

# Simulate first 12 months in detail
print("First 12 months detailed:")
print("-" * 80)
print(f"{'Month':>5} {'XGB Return':>12} {'Sentiment':>12} {'Total Return':>12} {'Price':>15} {'Cumul Return':>14}")
print("-" * 80)

for t in range(240):
    current_date = sim_world.index[t]
    sim_row = sim_world.iloc[[t]]
    current_hist = pd.concat([current_hist, sim_row])
    
    # Update features
    tail = current_hist.iloc[-24:].copy()
    tail = simulator._update_lags_and_deltas(tail)
    X_row = tail.iloc[[-1]][simulator.feature_columns]
    X_row = X_row.fillna(0).replace([np.inf, -np.inf], 0)
    
    # XGBoost prediction
    rational_log_return = float(simulator.xgb_model.predict(X_row)[0])
    
    # Sentiment
    monthly_shock = np.random.normal(simulator.sentiment_shock_mean, simulator.sentiment_shock_std)
    sentiment_score = (sentiment_score * simulator.sentiment_mean_reversion) + monthly_shock
    
    # Combined
    pred_log_return = rational_log_return + sentiment_score
    
    # Update price
    current_log_price = current_log_price + pred_log_return
    current_price = np.exp(current_log_price)
    
    # Store
    log_returns_xgb.append(rational_log_return)
    log_returns_with_sentiment.append(pred_log_return)
    sentiment_scores.append(sentiment_score)
    prices.append(current_price)
    
    # Write back to history
    current_hist.at[current_date, simulator.price_col] = pred_log_return
    current_hist.at[current_date, 'Log_Price'] = current_log_price
    
    # Print first 12 months
    if t < 12:
        cumul_return = np.exp(current_log_price) / anchor_price - 1
        print(f"{t+1:>5} {rational_log_return:>12.6f} {sentiment_score:>12.6f} {pred_log_return:>12.6f} ${current_price:>14,.2f} {cumul_return:>13.2%}")

print()
print("=" * 80)
print("STATISTICS OVER 240 MONTHS:")
print("=" * 80)

log_returns_xgb = np.array(log_returns_xgb)
log_returns_with_sentiment = np.array(log_returns_with_sentiment)
sentiment_scores = np.array(sentiment_scores)

print(f"XGBoost predictions:")
print(f"  Mean: {log_returns_xgb.mean():.6f} ({log_returns_xgb.mean()*12*100:.2f}% annualized)")
print(f"  Std: {log_returns_xgb.std():.6f}")
print(f"  Min: {log_returns_xgb.min():.6f}")
print(f"  Max: {log_returns_xgb.max():.6f}")
print()

print(f"Sentiment scores:")
print(f"  Mean: {sentiment_scores.mean():.6f}")
print(f"  Std: {sentiment_scores.std():.6f}")
print(f"  Min: {sentiment_scores.min():.6f}")
print(f"  Max: {sentiment_scores.max():.6f}")
print()

print(f"Combined (XGB + Sentiment):")
print(f"  Mean: {log_returns_with_sentiment.mean():.6f} ({log_returns_with_sentiment.mean()*12*100:.2f}% annualized)")
print(f"  Std: {log_returns_with_sentiment.std():.6f}")
print(f"  Min: {log_returns_with_sentiment.min():.6f}")
print(f"  Max: {log_returns_with_sentiment.max():.6f}")
print()

# Compare to reality
actual_prices = []
log_price = np.log(anchor_price)
for idx, row in test_df.iterrows():
    log_return = row['Log_Return_MoM']
    log_price += log_return
    actual_prices.append(np.exp(log_price))

final_predicted = prices[-1]
final_actual = actual_prices[-1]
error_pct = (final_predicted / final_actual - 1) * 100

print("=" * 80)
print("FINAL COMPARISON (20 years later):")
print("=" * 80)
print(f"Starting price: ${anchor_price:,.2f}")
print(f"Predicted final: ${final_predicted:,.2f}")
print(f"Actual final: ${final_actual:,.2f}")
print(f"Error: {error_pct:+.1f}%")
print()

# Calculate implied CAGR
pred_cagr = (final_predicted / anchor_price) ** (1/20) - 1
actual_cagr = (final_actual / anchor_price) ** (1/20) - 1
print(f"Predicted CAGR: {pred_cagr*100:.2f}%")
print(f"Actual CAGR: {actual_cagr*100:.2f}%")
print(f"CAGR error: {(pred_cagr - actual_cagr)*100:+.2f} percentage points")
print()

# Check if XGBoost is the problem
print("=" * 80)
print("ROOT CAUSE ANALYSIS:")
print("=" * 80)

# Theoretical price if just using mean XGB return
theoretical_price_xgb = anchor_price * np.exp(log_returns_xgb.sum())
theoretical_cagr_xgb = (theoretical_price_xgb / anchor_price) ** (1/20) - 1

print(f"If using ONLY XGBoost predictions (no sentiment):")
print(f"  Final price: ${theoretical_price_xgb:,.2f}")
print(f"  CAGR: {theoretical_cagr_xgb*100:.2f}%")
print(f"  Error vs actual: {(theoretical_price_xgb / final_actual - 1)*100:+.1f}%")
print()

# What if XGBoost was unbiased (mean = actual mean)?
actual_mean_return = test_df['Log_Return_MoM'].mean()
print(f"Actual mean log return in test period: {actual_mean_return:.6f} ({actual_mean_return*12*100:.2f}% annualized)")
print(f"XGBoost mean prediction: {log_returns_xgb.mean():.6f} ({log_returns_xgb.mean()*12*100:.2f}% annualized)")
print(f"XGBoost bias: {(log_returns_xgb.mean() - actual_mean_return):.6f} ({(log_returns_xgb.mean() - actual_mean_return)*12*100:.2f}% annualized)")
print()

# Show what happens with different biases over 20 years
print("Impact of bias over 20 years:")
for bias in [0.001, 0.002, 0.003, 0.004, 0.005]:
    biased_price = anchor_price * np.exp(bias * 240)
    error = (biased_price / final_actual - 1) * 100
    print(f"  Bias of {bias:.3f}/month ({bias*12*100:.2f}%/year) → ${biased_price:,.0f} ({error:+.0f}% error)")
