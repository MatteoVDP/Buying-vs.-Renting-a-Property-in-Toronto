import pandas as pd
import numpy as np

df = pd.read_csv('data/dataset-MVP-Mar2025.csv')

# Clean column names
def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.strip()
        new_col = new_col.replace(' ', '_')
        new_col = new_col.replace('(', '')
        new_col = new_col.replace(')', '')
        new_col = new_col.replace('-', '_')
        new_col = new_col.replace('/', '_')
        new_col = new_col.replace('.', '')
        new_col = new_col.lower()
        new_cols.append(new_col)
    df.columns = new_cols
    return df

df = clean_col_names(df)

# Remove rows with all NaN values
df.dropna(how='all', inplace=True)

cols_to_smooth = [
    'market_price_target_average,_detached_single_family_homes',
    'national_pop',
    'provincial_pop',
    'municipal_pop',
    'national_gdp_real,_seasonally_adjusted',
    'provincial_gdp_real,_seasonally_adjusted',
    'national_gdp_nominal',
    'provincial_gdp_nominal',
    'gdp_per_cap_real',
    'most_recent_quarterly_gdp_%_change_extended',
    'median_income_per_household_in_toronto',
    'national_debt_to_gdp',
    'provincial_debt_to_gdp',
    'ontario_net_international_migration_monthly',
    'ontario_net_interprovincial_migration_monthly',
    'ontario_net_non_permanent_residents',
    'housing_starts_sfh,_monthly',
    'under_construction_sfh,_monthly',
    'completions__sfh,_monthly'
]

for col in cols_to_smooth:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear')
    else:
        print(f"Warning: Column '{col}' not found in dataframe.")

#modifying features to allow for XGB to best use them
df['Log_Price'] = np.log(df['market_price_target_average,_detached_single_family_homes'])

df['GDP_Growth_YoY'] = df['national_gdp_real,_seasonally_adjusted'].pct_change(12)
df['National_Pop_Growth_YoY'] = df['national_pop'].pct_change(12)
df['Provincial_Pop_Growth_YoY'] = df['provincial_pop'].pct_change(12)
df['Municipal_Pop_Growth_YoY'] = df['municipal_pop'].pct_change(12)
df['Inflation_Rate_YoY'] = df['cpi___national,_all_products'].pct_change(12)
df['Labour_Force_Growth_YoY'] = df['total_labour_force_size_people_15_or_over,_employed_or_actively_seeking_work'].pct_change(12)
df['Inflation_Rate'] = df['cpi___national,_core'].pct_change(12)
df['Income_Growth_YoY'] = df['median_income_per_household_in_toronto'].pct_change(12)

df['Migration_Rate'] = (df['ontario_net_interprovincial_migration_monthly'] + df['ontario_net_international_migration_monthly'] - df['ontario_net_non_permanent_residents'])/df['provincial_pop']
df['NPR_Rate'] = df['ontario_net_non_permanent_residents']/df['provincial_pop']
df['housing_starts_per_cap'] = df['housing_starts_sfh,_monthly']/df['municipal_pop']
df['under_construction_per_cap'] = df['under_construction_sfh,_monthly']/df['municipal_pop']
df['completions_per_cap'] = df['completions__sfh,_monthly']/df['municipal_pop']

df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)

df['Ratio_Current_1'] = df['market_price_target_average,_detached_single_family_homes'] / df['median_income_per_household_in_toronto']
df['Affordability_Ratio_Lag1'] = df['Ratio_Current_1'].shift(1)
df['Ratio_Current_12'] = df['market_price_target_average,_detached_single_family_homes'] / df['median_income_per_household_in_toronto']
df['Affordability_Ratio_Lag12'] = df['Ratio_Current_12'].shift(12)

#creating new feaetures for lags and deltas to provide more context for XBG model

# We use a list to store all new series, then concat at the end to avoid fragmentation
new_features = []

# PRICE GROUP: market_price..., Log_Price (Lags: 1, 3, 6, 12 | Deltas: 1, 12 Pct)
price_cols = ['market_price_target_average,_detached_single_family_homes', 'Log_Price']
for col in [c for c in price_cols if c in df.columns]:
    for m in [1, 3, 6, 12]:
        new_features.append(df[col].shift(m).rename(f"{col}_lag_{m}"))
    for m in [1, 12]:
        new_features.append(df[col].pct_change(m).rename(f"{col}_delta_{m}m_pct"))

# MACRO GROUP: GDP, Pop, Income, CPI (Lags: 12 | Deltas: 12 Pct)
macro_cols = [
    'national_pop', 'provincial_pop', 'municipal_pop',
    'national_gdp_real,_seasonally_adjusted', 'provincial_gdp_real,_seasonally_adjusted',
    'median_income_per_household_in_toronto', 'cpi___national,_all_products', 'cpi___national,_core'
]
for col in [c for c in macro_cols if c in df.columns]:
    new_features.append(df[col].shift(12).rename(f"{col}_lag_12"))
    new_features.append(df[col].pct_change(12).rename(f"{col}_delta_12m_pct"))

# FINANCIALS GROUP: Mortgage Rates, Bonds (Lags: 1, 3 | Deltas: 1, 3 Diff)
fin_cols = [
    '3_month_t_bill', '2y_bond', '5y_bond', '10y_bond', 
    'variable_mortgage_rate', '5_year_fixed_mortgage_rate', 
    '5_year_fixed_mortgage_qualifying_rate', 'yield_curve_slope'
]
for col in [c for c in fin_cols if c in df.columns]:
    for m in [1, 3]:
        new_features.append(df[col].shift(m).rename(f"{col}_lag_{m}"))
        new_features.append(df[col].diff(m).rename(f"{col}_delta_{m}m_diff"))

# SUPPLY GROUP: Starts, Under Construction, Volume (Lags: 12 | Deltas: 12 Pct)
supply_cols = [
    'housing_starts_sfh,_monthly', 'under_construction_sfh,_monthly', 
    'completions__sfh,_monthly', 'sales_volume'
]
for col in [c for c in supply_cols if c in df.columns]:
    new_features.append(df[col].shift(12).rename(f"{col}_lag_12"))
    new_features.append(df[col].pct_change(12).rename(f"{col}_delta_12m_pct"))

# MIGRATION GROUP: NPRs and International (Lags: 1, 12 | Deltas: 12 Pct)
migration_cols = [
    'ontario_net_international_migration_monthly', 
    'ontario_net_interprovincial_migration_monthly',
    'ontario_net_non_permanent_residents'
]
for col in [c for c in migration_cols if c in df.columns]:
    for m in [1, 12]:
        new_features.append(df[col].shift(m).rename(f"{col}_lag_{m}"))
    new_features.append(df[col].pct_change(12).rename(f"{col}_delta_12m_pct"))

# Join all new columns to the original dataframe at once
df = pd.concat([df] + new_features, axis=1)

# Define the list of columns to drop
columns_to_drop = [
    # Original target variable (we now have Log_Price)
    'market_price_target_average,_detached_single_family_homes',
    
    # Original columns used for Year-over-Year growth features
    'national_gdp_real,_seasonally_adjusted',
    'national_pop',
    'provincial_pop',
    'municipal_pop',
    'cpi___national,_all_products',
    'total_labour_force_size_people_15_or_over,_employed_or_actively_seeking_work',
    'id',
    'temp',
    'sales_volume',
    'year',
    'month',
    'provincial_gdp_real,_seasonally_adjusted',
    'national_gdp_nominal',
    'provincial_gdp_nominal',
    'gdp_per_cap_real',
    'national_debt_billions',
    'provincial_debt',
    'Ratio_Current_1',
    'Ratio_Current_12',
    'ontario_net_interprovincial_migration_monthly',
    'ontario_net_international_migration_monthly',
    'ontario_net_non_permanent_residents',
    'housing_starts_sfh,_monthly',
    'under_construction_sfh,_monthly',
    'completions__sfh,_monthly',
    'median_income_per_household_in_toronto',
    'cpi___national,_all_products', 
    'cpi___national,_core'
    
]

# Drop the specified columns from the dataframe
df.drop(columns=columns_to_drop, inplace=True, axis=1)

#drop indices that have NaNs from lag, delta, and average features
df.drop(df.index[0:12], inplace=True)

df.to_csv("data/processed_data.csv", index=True)
