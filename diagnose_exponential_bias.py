#!/usr/bin/env python3
"""
Comprehensive diagnostic script to identify root causes of exponential price growth.
Analyzes: XGBoost features, ARIMA trends, training data bias, affordability feedback,
SARIMAX coefficients, and log return biases.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/scripts')
from market_simulator import MarketSimulator
import warnings
warnings.filterwarnings('ignore')

def analyze_xgboost_feature_importance(simulator):
    """Identify which features most influence price predictions."""
    print("="*80)
    print("1. XGBoost Feature Importance Analysis")
    print("="*80)
    
    if simulator.xgb_model is None:
        print("❌ XGBoost model not fitted")
        return
    
    # Get feature importance
    importances = simulator.xgb_model.feature_importances_
    feature_names = simulator.xgb_model.feature_names_in_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop 15 Features Driving Price Predictions:")
    print(f"{'Rank':<6} {'Feature':<50} {'Importance':<10} {'Cumulative %'}")
    print("-"*90)
    
    cumulative = 0
    for i, idx in enumerate(indices[:15]):
        cumulative += importances[idx]
        print(f"{i+1:<6} {feature_names[idx]:<50} {importances[idx]:.4f}    {cumulative*100:>6.2f}%")
    
    # Check if any single feature dominates
    max_importance = importances[indices[0]]
    if max_importance > 0.3:
        print(f"\n⚠️  WARNING: Feature '{feature_names[indices[0]]}' has {max_importance*100:.1f}% importance")
        print(f"    This single feature may be driving exponential predictions!")
    
    return dict(zip(feature_names, importances))


def analyze_arima_trends(simulator):
    """Check if ARIMA models have embedded drift causing systematic upward bias."""
    print("\n" + "="*80)
    print("2. ARIMA Tier 1 Trend Analysis")
    print("="*80)
    
    drift_vars = []
    for var in simulator.tier1_vars:
        model = simulator.arima_models.get(var)
        if model is None:
            continue
        
        # Check model order
        order = model.order
        print(f"\n{var}:")
        print(f"  Order: ARIMA{order}")
        
        # Simulate forward to check for drift
        try:
            forecast = model.predict(n_periods=120)  # 10 years
            mean_change_per_step = (forecast[-1] - forecast[0]) / 120
            
            # Get historical mean change
            hist_data = simulator.df[var].dropna()
            hist_diff = hist_data.diff().dropna()
            hist_mean_change = hist_diff.mean()
            
            print(f"  Historical mean monthly change: {hist_mean_change:.6f}")
            print(f"  Forecast mean monthly change:   {mean_change_per_step:.6f}")
            
            # Check if forecast drift is significantly larger than historical
            if abs(mean_change_per_step) > abs(hist_mean_change) * 2:
                drift_vars.append(var)
                print(f"  ⚠️  Forecast drift is {abs(mean_change_per_step/hist_mean_change):.1f}x historical!")
        except Exception as e:
            print(f"  ❌ Forecast failed: {str(e)[:50]}")
    
    if drift_vars:
        print(f"\n⚠️  Variables with excessive drift: {drift_vars}")
        print("    These may be causing exponential growth through exogenous propagation!")
    else:
        print("\n✓ No excessive drift detected in Tier 1 ARIMA models")


def analyze_training_data_bias(simulator):
    """Check if recent historical data has unusual trends that bias the model."""
    print("\n" + "="*80)
    print("3. Training Data Bias Analysis")
    print("="*80)
    
    df = simulator.df
    target = 'Log_Return_MoM'
    
    if target not in df.columns:
        print(f"❌ Target '{target}' not found in data")
        return
    
    returns = df[target].dropna()
    
    # Analyze different time periods
    periods = {
        'Full history': returns,
        'Last 5 years': returns[-60:],
        'Last 3 years': returns[-36:],
        'Last 1 year': returns[-12:],
    }
    
    print(f"\nTarget Variable: {target}")
    print(f"{'Period':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)
    
    for period_name, data in periods.items():
        if len(data) > 0:
            print(f"{period_name:<20} {data.mean():<12.6f} {data.std():<12.6f} "
                  f"{data.min():<12.6f} {data.max():<12.6f}")
    
    # Check if recent mean is significantly positive
    recent_mean = returns[-12:].mean()
    historical_mean = returns.mean()
    
    print(f"\nRecent 12-month mean: {recent_mean:.6f}")
    print(f"Full historical mean: {historical_mean:.6f}")
    
    if recent_mean > historical_mean * 2:
        print(f"⚠️  Recent returns are {recent_mean/historical_mean:.1f}x higher than historical!")
        print("    XGBoost may be learning to predict persistent positive returns!")
    elif recent_mean > 0.01:  # 1% monthly = 12.7% annually compounded
        print(f"⚠️  Recent returns average {recent_mean*100:.2f}% per month")
        print(f"    This would compound to {((1+recent_mean)**12 - 1)*100:.1f}% annually!")
        print("    If XGBoost extrapolates this trend, exponential growth is inevitable!")
    else:
        print("✓ Recent returns do not appear excessively biased")


def analyze_affordability_feedback(simulator):
    """Check if affordability feedback loop amplifies predictions."""
    print("\n" + "="*80)
    print("4. Affordability Feedback Loop Analysis")
    print("="*80)
    
    df = simulator.df
    
    # Check if affordability is used as a feature
    affordability_features = [col for col in df.columns if 'affordability' in col.lower() or 
                             'price_to_income' in col.lower() or 'pti' in col.lower()]
    
    if not affordability_features:
        print("✓ No affordability features detected - no feedback loop risk")
        return
    
    print(f"Found affordability features: {affordability_features}")
    
    # Analyze correlation with target
    target = 'Log_Return_MoM'
    if target in df.columns:
        print(f"\nCorrelation with {target}:")
        for feat in affordability_features:
            if feat in df.columns:
                corr = df[feat].corr(df[target])
                print(f"  {feat}: {corr:.4f}")
                
                if abs(corr) > 0.5:
                    print(f"    ⚠️  Strong correlation! May create feedback amplification!")
    
    # Check if affordability is computed during forecasting
    print("\n⚠️  WARNING: If affordability depends on forecasted prices,")
    print("    and prices depend on affordability, this creates exponential feedback!")
    print("    Recommendation: Remove affordability from exogenous features!")


def analyze_sarimax_coefficients(simulator):
    """Inspect SARIMAX coefficients to check if exogenous relationships are reasonable."""
    print("\n" + "="*80)
    print("5. SARIMAX Exogenous Coefficient Analysis")
    print("="*80)
    
    suspicious_vars = []
    
    for tier, vars_list in [('Tier 2', simulator.tier2_vars), ('Tier 3', simulator.tier3_vars)]:
        print(f"\n{tier}:")
        for var in vars_list:
            model = simulator.sarimax_models.get(var)
            if model is None:
                continue
            
            try:
                # Get parameter estimates
                params = model.params
                exog_params = [p for p in params.index if p not in ['ar.L1', 'ma.L1', 'sigma2', 
                                                                      'intercept', 'drift']]
                
                if len(exog_params) == 0:
                    continue
                
                print(f"\n  {var}:")
                # Check for unreasonably large coefficients
                for param_name in exog_params[:5]:  # Top 5
                    coef = params[param_name]
                    if abs(coef) > 10:
                        print(f"    ⚠️  {param_name}: {coef:.4f} (LARGE!)")
                        suspicious_vars.append((var, param_name, coef))
                    else:
                        print(f"    {param_name}: {coef:.4f}")
            except Exception as e:
                print(f"  ❌ {var}: {str(e)[:50]}")
    
    if suspicious_vars:
        print(f"\n⚠️  Found {len(suspicious_vars)} suspiciously large exogenous coefficients!")
        print("    Large coefficients can amplify noise from Tier 1 into exponential growth!")


def analyze_log_return_predictions(simulator):
    """Check if XGBoost systematically predicts positive log returns."""
    print("\n" + "="*80)
    print("6. XGBoost Log Return Prediction Bias")
    print("="*80)
    
    if simulator.xgb_model is None:
        print("❌ XGBoost model not fitted")
        return
    
    # Get training predictions
    df = simulator.df
    target = 'Log_Return_MoM'
    
    if target not in df.columns:
        print(f"❌ Target '{target}' not in data")
        return
    
    # Get feature matrix
    non_target_cols = [c for c in df.columns if c != target and c != 'Date']
    X = df[non_target_cols].fillna(0)
    y_true = df[target].fillna(0)
    
    # Predict on training data
    y_pred = simulator.xgb_model.predict(X)
    
    print(f"\nTraining Set Predictions:")
    print(f"  True mean:      {y_true.mean():.6f}")
    print(f"  Predicted mean: {y_pred.mean():.6f}")
    print(f"  True std:       {y_true.std():.6f}")
    print(f"  Predicted std:  {y_pred.std():.6f}")
    
    # Check bias
    bias = y_pred.mean() - y_true.mean()
    print(f"\nPrediction bias: {bias:.6f}")
    
    if y_pred.mean() > 0.005:  # > 0.5% monthly
        print(f"⚠️  XGBoost predicts average monthly return of {y_pred.mean()*100:.3f}%")
        print(f"    This compounds to {((1+y_pred.mean())**12 - 1)*100:.1f}% annually!")
        print(f"    Over 25 years, this causes {((1+y_pred.mean())**300 - 1)*100:.1f}% growth!")
    
    # Check if predictions are bounded
    print(f"\nPrediction range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"True range:       [{y_true.min():.6f}, {y_true.max():.6f}]")
    
    if y_pred.max() > y_true.max() * 1.5:
        print("⚠️  XGBoost predicts returns larger than historical maximum!")


def main():
    """Run comprehensive diagnostic analysis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPONENTIAL BIAS DIAGNOSTIC")
    print("="*80)
    
    # Load data and fit model
    print("\nLoading data and fitting models...")
    df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
    
    # Initialize simulator (tier1_noise_scale and black_swan_prob are set in __init__)
    simulator = MarketSimulator(df=df)
    
    print("Fitting models (this may take a minute)...")
    simulator.fit()
    
    # Run diagnostics
    analyze_xgboost_feature_importance(simulator)
    analyze_arima_trends(simulator)
    analyze_training_data_bias(simulator)
    analyze_affordability_feedback(simulator)
    analyze_sarimax_coefficients(simulator)
    analyze_log_return_predictions(simulator)
    
    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
Based on the diagnostics above, likely root causes of exponential growth:

1. If recent training data shows high positive returns:
   → Retrain on longer historical period to reduce recency bias
   → Consider detrending the target variable
   → Add temporal cross-validation to prevent overfitting to recent trends

2. If XGBoost predicts systematically positive returns:
   → Increase regularization (max_depth, min_child_weight, gamma)
   → Add a constraint that mean prediction should be near historical mean
   → Consider using quantile regression instead of point estimates

3. If ARIMA models have excessive drift:
   → Use differencing (d=1) to remove trend
   → Consider forcing models to have zero mean drift
   → Use seasonal decomposition to separate trend from cycles

4. If affordability creates feedback:
   → Remove affordability features that depend on forecasted prices
   → Use lagged affordability only (breaks feedback loop)
   → Consider making affordability exogenous to the system

5. If SARIMAX coefficients are too large:
   → Regularize SARIMAX (add penalty to exogenous coefficients)
   → Standardize exogenous inputs before fitting
   → Remove exogenous variables with coefficients > 10

6. If feature importance is concentrated:
   → Check what that dominant feature represents
   → If it's recursive (depends on predictions), remove it
   → Add feature engineering to create more balanced predictors
    """)

if __name__ == "__main__":
    main()
