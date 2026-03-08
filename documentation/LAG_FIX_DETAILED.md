# LAG/DELTA/RA "GARBAGE VALUES" FIX

## Problem Description

After implementing the initial fix to save lag/delta/RA columns, the values were present but **incorrect**. Specifically:

- All lag_6 values were identical (e.g., 0.029187) across all 300 forecast months
- Lag values were not actually referencing the correct historical month
- The columns had values but they were "garbage" - not proper lags at all

## Root Cause Analysis

### The Bug

The issue was in how we were writing calculated features back to `current_hist`:

```python
# OLD CODE (BUGGY):
tail = current_hist.iloc[start_idx:].copy()
tail = self._update_lags_and_deltas(tail)

# Write ALL calculated features for ALL rows in tail
for col in tail.columns:
    current_hist.loc[tail.index, col] = tail[col]  # ❌ OVERWRITES EVERYTHING
```

### Why This Caused Garbage Values 

At each forecast iteration `t`, we:
1. Took the last 50 rows of `current_hist` (including previous forecast months)
2. Calculated lags on this entire 50-row window
3. **Overwrote all 50 rows' lag values** with the newly calculated values

This meant:
- **Month 0**: Calculate lag_6 for month 0 (references historical data)
- **Month 6**: Recalculate lag_6 for entire tail → **OVERWRITES Month 0's lag_6** with a new calculation
- **Month 12**: Recalculate again → **OVERWRITES everyone's lag_6** again

So earlier forecast months' lag values kept getting overwritten, eventually converging to some stable but incorrect value.

## The Fix

Only write back the **current row's** calculated features, not the entire tail:

```python
# NEW CODE (FIXED):
tail = current_hist.iloc[start_idx:].copy()
tail = self._update_lags_and_deltas(tail)

# Write ONLY the current (last) row's calculated features
current_row_features = tail.iloc[-1]
for col in tail.columns:
    if any(tag in col for tag in ['_lag_', '_delta_', '_RA_']):
        # Ensure column exists
        if col not in current_hist.columns:
            current_hist[col] = np.nan
        current_hist.at[current_date, col] = current_row_features[col]  # ✓ ONLY CURRENT ROW
```

### Why This Works

Now at each iteration:
- **Month 0**: Calculate and save lag_6 for month 0 → **NEVER TOUCHED AGAIN**
- **Month 1**: Calculate and save lag_6 for month 1 → **NEVER TOUCHED AGAIN**- **Month 6**: Calculate and save lag_6 for month 6 → **Previous months' lags remain intact**

Each forecast month's derived features are calculated once and preserved.

## Files Modified

1. **scripts/market_simulator.py** (lines ~454-468)
   - Fixed the forecast loop to only write current row's features

2. **scripts/audit_updated.py** (lines ~125-137)
   - Fixed the audit script's forecast loop identically

## Verification

### Quick Check (on existing forecast)
```bash
python quick_diagnostic.py
```

Expected output:
```
✓ OK - multiple unique values
Range: 0.001234 to 0.045678
```

Instead of:
```
❌ BUG - all 300 values are 0.029187
```

### Full Test (12-month simulation)
```bash
python test_lag_fix.py
```

Expected output:
```
✅ SUCCESS! Lags are updating correctly month-to-month
Results: 11 matches, 0 mismatches
✅ FIXED: lag_6 values vary
```

### Generate New Forecast
```bash
python scripts/audit_updated.py
```

Then check:
```bash
python quick_diagnostic.py
```

## Technical Details

### How Lags Should Work

For `Income_Growth_YoY_lag_1` at forecast month `t`:
- Value should equal `Income_Growth_YoY` at month `t-1`
- This is calculated by: `tail['Income_Growth_YoY'].shift(1)`
- The shift operation references the row **1 position earlier in the tail**

### Why the Tail Window

We use a 50-row tail window to ensure:
- lag_1 has 1 row of history
- lag_6 has 6 rows of history
- lag_12 has 12 rows of history
- lag_24 has 24 rows of history
- Rolling averages (RA_12, RA_24) have sufficient window

### Column Initialization

The fix ensures columns are properly initialized:
```python
if col not in current_hist.columns:
    current_hist[col] = np.nan  # Initialize column with NaN
```

This prevents KeyError when writing to a column that doesn't exist yet.

## Expected Behavior After Fix

When examining the forecast CSV:

**BEFORE FIX (GARBAGE):**
```
Date          Income_Growth  lag_6     lag_12
2025-04-01    0.033653      0.029187  0.027678
2025-05-01   -0.004384      0.029187  0.027678  ← Same value
2025-06-01    0.025725      0.029187  0.027678  ← Same value
```

**AFTER FIX (CORRECT):**
```
Date          Income_Growth  lag_6     lag_12
2025-04-01    0.033653      0.029187  0.027678
2025-05-01   -0.004384      0.025341  0.027321
2025-06-01    0.025725      0.031256  0.028119
```

Note: Values will vary based on the simulated exogenous variables, but they should NOT all be identical.

## Implementation Notes

1. **Performance**: Only writing one row instead of 50 per iteration is also slightly more efficient

2. **Accuracy**: This fix is critical for model accuracy. Using correct lag values means the XGBoost model gets proper historical context for each prediction.

3. **Affordability Ratio**: This column was already updating correctly (calculated after prediction) and is unaffected by this fix.

4. **Delta & RA Columns**: The same bug affected deltas and rolling averages. The fix applies to all derived feature types.
