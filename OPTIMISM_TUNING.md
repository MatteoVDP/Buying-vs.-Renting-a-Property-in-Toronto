"""
GUIDE: ADJUSTING MODEL OPTIMISM IN MARKETSIMULATOR
===================================================

Two types of changes made:
1. Doomsday probability increased (1 every 200 years instead of 8,333)
2. Sentiment mechanism made configurable for optimism tuning

This document explains how to control model optimism.
"""

# ==============================================================================
# PART 1: DOOMSDAY PROBABILITY ADJUSTMENT
# ==============================================================================

"""
CHANGE MADE:
    OLD: self.doomsday_prob = 0.00001      # Every 8,333 years
    NEW: self.doomsday_prob = 0.0004167    # Every 200 years (~1 magnitude more likely)

CALCULATION:
    Expected time to event (in months) = 1 / probability
    
    For every 200 years:
        200 years × 12 months/year = 2,400 months
        Probability = 1 / 2,400 = 0.0004167 per month
    
    Compare:
        - Old: 1 / 100,000 = 0.00001 per month (expects ~1 event per 8,333 years)
        - New: 1 / 2,400 = 0.0004167 per month (expects ~1 event per 200 years)
        - Ratio: 41.67x more likely

IMPACT:
    In a 100-month forecast window:
        - Old: P(at least 1 doomsday) ≈ 0.1%
        - New: P(at least 1 doomsday) ≈ 4.1%
    
    In 50 iterations of 100 months:
        - Old: ~0 expected doomsday events
        - New: ~2 expected doomsday events (you should see them occasionally)

TO FURTHER ADJUST:
    - Make even MORE likely: 0.001 (every 80 years)
    - Make less likely: 0.0001 (every 833 years)
    - Turn off: 0.0 (no doomsday events)
"""

# ==============================================================================
# PART 2: MARKET SENTIMENT CONFIGURATION (The Optimism Lever)
# ==============================================================================

"""
NEW PARAMETERS ADDED:

    self.sentiment_shock_mean = 0.0
        └─ OPTIMISM CONTROL: Mean of monthly sentiment shocks
           - 0.0 = NEUTRAL (no bias)
           - 0.005 = OPTIMISTIC (slight upward bias)
           - 0.01 = VERY OPTIMISTIC (strong upward bias)
           - -0.005 = PESSIMISTIC (downward bias)

    self.sentiment_shock_std = 0.02
        └─ VOLATILITY CONTROL: Std dev of sentiment shocks
           - 0.02 = NORMAL (current, realistic)
           - 0.01 = LOW (calmer market, fewer sentiment swings)
           - 0.03 = HIGH (more volatile sentiment)

    self.sentiment_mean_reversion = 0.95
        └─ RECOVERY SPEED: How fast sentiment mean-reverts
           - 0.95 = NORMAL (95% decay per month, gentle recovery)
           - 0.97 = FAST (faster recovery from downturns → more optimistic)
           - 0.93 = SLOW (slower recovery from downturns → more pessimistic)

HOW SENTIMENT WORKS (Simplified):
    
    Each month:
        1. Generate random shock: shock = N(mean, std)
        2. Update sentiment: sentiment = sentiment * decay + shock
        3. Apply to price: return = rational_return + sentiment
    
    Example with NEUTRAL settings (current):
        Month 1: sentiment = 0.0 + 0.005 = 0.005 (small positive shock)
        Month 2: sentiment = 0.005 * 0.95 + (-0.002) = 0.0025
        Month 3: sentiment = 0.0025 * 0.95 + 0.001 = 0.0034
        → Sentiment oscillates around 0, creating bubbles and crashes
    
    Example with OPTIMISTIC settings:
        (mean=0.01, std=0.015, decay=0.97)
        Month 1: sentiment = 0.0 + 0.012 = 0.012 (positive shock)
        Month 2: sentiment = 0.012 * 0.97 + 0.015 = 0.0271 (accumulates!)
        Month 3: sentiment = 0.0271 * 0.97 + 0.008 = 0.0354 (keeps growing)
        → Sentiment drifts upward → sustained price optimism
"""

# ==============================================================================
# PART 3: PRACTICAL OPTIMISM SCENARIOS
# ==============================================================================

"""
SCENARIO A: NEUTRAL (Current Default)
────────────────────────────────────

    simulator.sentiment_shock_mean = 0.0
    simulator.sentiment_shock_std = 0.02
    simulator.sentiment_mean_reversion = 0.95

Use case: Unbiased forecasts, realistic volatility
Result: Market oscillates naturally, no systematic bias


SCENARIO B: SLIGHTLY OPTIMISTIC (Recommended for "Base Case")
────────────────────────────────────

    simulator.sentiment_shock_mean = 0.005
    simulator.sentiment_shock_std = 0.02
    simulator.sentiment_mean_reversion = 0.96

How it increases optimism:
    • +0.5% monthly sentiment bias
    • Faster recovery from downturns (decay 0.96 vs 0.95)
    • Creates ~2-3% annualized pink noise upward drift

Intuition:
    Toronto's fundamentals (immigration, economic strength) support
    modest optimism. This captures that realistic positive drift.


SCENARIO C: MODERATELY OPTIMISTIC (Bull Market Scenario)
────────────────────────────────────

    simulator.sentiment_shock_mean = 0.01
    simulator.sentiment_shock_std = 0.015
    simulator.sentiment_mean_reversion = 0.97

How it increases optimism:
    • +1% monthly sentiment bias (compounding!)
    • Much faster mean-reversion (0.97 vs 0.95)
    • Lower shock volatility (0.015 vs 0.02)
    • Creates ~12%+ annualized upward drift

Result: Prices climb steadily with shallow dips
         Good for "best case" scenarios


SCENARIO D: VERY OPTIMISTIC (Bull Market Euphoria)
────────────────────────────────────

    simulator.sentiment_shock_mean = 0.015
    simulator.sentiment_shock_std = 0.01
    simulator.sentiment_mean_reversion = 0.97

How it increases optimism:
    • +1.5% monthly sentiment bias
    • Minimal downside volatility
    • Very fast recovery

Result: Strong upward trend with rare pullbacks
        Use case: Testing "what if Toronto becomes like Vancouver/Toronto mega-boom"
        Risk: Unrealistic (ignores market fundamentals)


SCENARIO E: PESSIMISTIC (Bear Market)
────────────────────────────────────

    simulator.sentiment_shock_mean = -0.01
    simulator.sentiment_shock_std = 0.025
    simulator.sentiment_mean_reversion = 0.93

How it decreases optimism:
    • -1% monthly sentiment bias
    • Slower recovery from crashes (0.93 vs 0.95)
    • Higher volatility

Result: Prices struggle to gain traction, pull-back heavy
        Use case: Stress testing, recession scenarios
"""

# ==============================================================================
# PART 4: PRACTICAL USAGE EXAMPLES
# ==============================================================================

"""
EXAMPLE 1: Run Optimistic Backtest
──────────────────────────────────

    from market_simulator import MarketSimulator
    import pandas as pd
    
    df = pd.read_csv('data/processed_data.csv')
    train_df = df.iloc[:-100]
    
    # Initialize simulator
    simulator = MarketSimulator(train_df, seed=42, start_market_price=777181)
    
    # SET OPTIMISTIC PARAMETERS
    simulator.sentiment_shock_mean = 0.005      # Slight optimism
    simulator.sentiment_shock_std = 0.02
    simulator.sentiment_mean_reversion = 0.96   # Faster recovery
    
    # Train
    simulator.fit(train_df)
    
    # Forecast
    paths = simulator.forecast_price(iterations=50, steps=100)
    median = paths.median(axis=1)
    
    print(f"Optimistic forecast: {median.iloc[-1]:.0f}")
    # Expect: Higher median price than neutral or pessimistic runs


EXAMPLE 2: Sensitivity Analysis - Compare 3 Scenarios
──────────────────────────────────────────────────────

    scenarios = {
        'Pessimistic': {'mean': -0.01, 'std': 0.025, 'decay': 0.93},
        'Neutral': {'mean': 0.0, 'std': 0.02, 'decay': 0.95},
        'Optimistic': {'mean': 0.01, 'std': 0.015, 'decay': 0.97},
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        simulator = MarketSimulator(train_df, seed=42, start_market_price=777181)
        
        # Apply scenario parameters
        simulator.sentiment_shock_mean = params['mean']
        simulator.sentiment_shock_std = params['std']
        simulator.sentiment_mean_reversion = params['decay']
        
        simulator.fit(train_df)
        paths = simulator.forecast_price(iterations=50, steps=100)
        
        final_price = paths.median(axis=1).iloc[-1]
        results[scenario_name] = final_price
        
        print(f"{scenario_name:15} → ${final_price:,.0f}")
    
    # Expected output:
    # Pessimistic     → $850,000
    # Neutral         → $920,000
    # Optimistic      → $1,000,000


EXAMPLE 3: Find Break-Even Parameters
──────────────────────────────────────
If you want prices to increase by exactly 3% annually:

    # 3% annual ≈ 0.247% monthly
    # Requires: sentiment_shock_mean ≈ 0.002
    
    simulator.sentiment_shock_mean = 0.002
    simulator.sentiment_shock_std = 0.02
    simulator.sentiment_mean_reversion = 0.95
    
    paths = simulator.forecast_price(iterations=50, steps=100)
    # Verify: (paths.iloc[-1] / paths.iloc[0]) ** (12/100) - 1 ≈ 0.03
"""

# ==============================================================================
# PART 5: MATHEMATICAL EXPLANATION
# ==============================================================================

"""
SENTIMENT DYNAMICS:
    s(t+1) = decay * s(t) + N(mean, std²)

EQUILIBRIUM SENTIMENT:
    In steady state: E[s] = mean / (1 - decay)
    
    Neutral (mean=0, decay=0.95): E[s] = 0 / 0.05 = 0
    Slightly optimistic (mean=0.005, decay=0.96): E[s] = 0.005 / 0.04 = 0.125
    Moderately optimistic (mean=0.01, decay=0.97): E[s] = 0.01 / 0.03 = 0.333

VOLATILITY OF SENTIMENT:
    Var(s) = std² / (1 - decay²)
    
    Higher decay → Lower variance → Smoother sentiment → Less boom/bust

IMPACT ON ANNUAL RETURNS:
    Rough approximation:
        Annual return ≈ fundamental_return + 12 * sentiment_shock_mean
    
    Neutral: annual return ≈ fundamental + 0%
    Optimistic (mean=0.005): annual return ≈ fundamental + 6%
    Very optimistic (mean=0.01): annual return ≈ fundamental + 12%

Note: These are SENTIMENTS. The actual XGBoost model predicts fundamentals.
      Sentiment is an OVERLAY that represents market psychology & bubbles.
"""

# ==============================================================================
# PART 6: RECOMMENDATIONS
# ==============================================================================

"""
DEFAULT (from now on):
    • Doomsday: 1 every 200 years (0.0004167)
    • Sentiment: Neutral (0.0 mean, 0.02 std, 0.95 decay)

FOR OFFICIAL FORECASTS:
    Use "Slightly Optimistic":
        • Sentiment mean: 0.005 (captures Toronto's positive fundamentals)
        • Decay: 0.96 (reasonable mean-reversion)
        • Volatility: 0.02 (realistic)

FOR SCENARIO ANALYSIS:
    • Bull case: sentiment_mean = 0.01
    • Base case: sentiment_mean = 0.005
    • Bear case: sentiment_mean = -0.005

FOR STRESS TESTING:
    • Run with doomsday_prob = 0.001 (10x more likely for testing)
    • Use pessimistic sentiment to see worst-case unrolling

NEVER (breaks the model):
    • Sentiment decay < 0.9 (no mean-reversion)
    • Sentiment decay > 1.0 (sentiment never stabilizes, explodes)
    • Shock std < 0 (nonsensical)
    • Shock mean > 0.02 (unrealistically optimistic, ~12% annual drift)
"""

print(__doc__)
