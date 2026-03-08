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
    'sales_volume',
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
df['Log_Return_MoM'] = df['Log_Price'].diff()
df['sales_volume_MoM'] = df['sales_volume'].pct_change(1)
df['sales_volume_YoY'] = df['sales_volume'].pct_change(12)

df['GDP_Growth_YoY'] = df['national_gdp_real,_seasonally_adjusted'].pct_change(12)
df['National_Pop_Growth_YoY'] = df['national_pop'].pct_change(12)
#df['Provincial_Pop_Growth_YoY'] = df['provincial_pop'].pct_change(12)
df['Municipal_Pop_Growth_YoY'] = df['municipal_pop'].pct_change(12)
df['Inflation_Rate_MoM'] = df['cpi___national,_all_products'].pct_change(1)
df['Inflation_Rate_YoY'] = df['cpi___national,_all_products'].pct_change(12)
df['Labour_Force_Growth_YoY'] = df['total_labour_force_size_people_15_or_over,_employed_or_actively_seeking_work'].pct_change(12)
df['Income_Growth_YoY'] = df['median_income_per_household_in_toronto'].pct_change(12)

df['Migration_Rate'] = (df['ontario_net_interprovincial_migration_monthly'] + df['ontario_net_international_migration_monthly'] - df['ontario_net_non_permanent_residents'])/df['provincial_pop']
df['NPR_Rate'] = df['ontario_net_non_permanent_residents']/df['provincial_pop']
df['housing_starts_per_cap'] = df['housing_starts_sfh,_monthly']/df['municipal_pop']
df['under_construction_per_cap'] = df['under_construction_sfh,_monthly']/df['municipal_pop']
df['completions_per_cap'] = df['completions__sfh,_monthly']/df['municipal_pop']

df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)

df['Affordability_Ratio'] = df['market_price_target_average,_detached_single_family_homes'] / df['median_income_per_household_in_toronto']
df.columns

new_features = []

fe_rates = ['3_month_t_bill', '5y_bond',
       'yield_curve_slope', 'variable_mortgage_rate', '5_year_fixed_mortgage_qualifying_rate'] # delta 1, 3, 6, 12

fe_YoY = ['GDP_Growth_YoY',
       'National_Pop_Growth_YoY', #'Provincial_Pop_Growth_YoY',
       'Municipal_Pop_Growth_YoY', 'Inflation_Rate_YoY',
       'Labour_Force_Growth_YoY', 'Income_Growth_YoY'] #lag 6 - RA 12
       
fe_immigration = ['Migration_Rate', 'NPR_Rate'] #lag 6, 12 - delta 6, 12 - RA 12

fe_target = ['Log_Return_MoM'] #lag 1, 3, 6, 12 TARGET

fe_sales_volume = ['sales_volume_MoM', 'sales_volume_YoY'] #lag 1, 3, 12 DROP

fe_affordability = ['Affordability_Ratio'] #lag 6, 12 - delta 6, 12 - RA 12

fe_supply = ['housing_starts_per_cap', 'under_construction_per_cap', 'completions_per_cap'] #lag 12, 24 - RA 6, 12

for col in [c for c in fe_rates if c in df.columns]:
    for m in [1, 3, 6, 12]:
        new_features.append(df[col].diff(m).rename(f"{col}_delta_{m}"))

for col in [c for c in fe_YoY if c in df.columns]:
    new_features.append(df[col].shift(6).rename(f"{col}_lag_6"))
    new_features.append(df[col].rolling(window=12).mean().rename(f"{col}_RA_12"))

for col in [c for c in fe_immigration if c in df.columns]:
    new_features.append(df[col].rolling(window=12).mean().rename(f"{col}_RA_12"))
    for m in [6, 12]:
        new_features.append(df[col].shift(m).rename(f"{col}_lag_{m}"))
        new_features.append(df[col].diff(m).rename(f"{col}_delta_{m}"))

for col in [c for c in fe_target if c in df.columns]:
    for m in [1, 3, 6, 12]:
        new_features.append(df[col].shift(m).rename(f"{col}_lag_{m}"))

for col in [c for c in fe_sales_volume if c in df.columns]:
    for m in [1, 3, 12]:
        new_features.append(df[col].shift(m).rename(f"{col}_lag_{m}"))

for col in [c for c in fe_affordability if c in df.columns]:
    new_features.append(df[col].rolling(window=12).mean().rename(f"{col}_RA_12"))
    new_features.append(df[col].rolling(window=24).mean().rename(f"{col}_RA_24"))
    for m in [3, 12]:
        new_features.append(df[col].shift(m).rename(f"{col}_lag_{m}"))
        new_features.append(df[col].diff(m).rename(f"{col}_delta_{m}"))

for col in [c for c in fe_supply if c in df.columns]:
    new_features.append(df[col].shift(12).rename(f"{col}_lag_12"))
    new_features.append(df[col].shift(24).rename(f"{col}_lag_24"))
    new_features.append(df[col].rolling(window=6).mean().rename(f"{col}_RA_6"))
    new_features.append(df[col].rolling(window=12).mean().rename(f"{col}_RA_12"))

# --- 3. MERGE & CLEANUP ---
# Concat all new features
df = pd.concat([df] + new_features, axis=1)

# CRITICAL: Replace Infinities from pct_change with 0
# This prevents XGBoost from crashing later
df.replace([np.inf, -np.inf], 0, inplace=True)

# Optional: Fill NaNs created by lags with 0 or drop (Simulator usually handles drops, but filling 0 is safer for feature engineering)
# df.fillna(0, inplace=True)

# Define the list of columns to drop
columns_to_drop = [
    # Original target variable (we now use Log_Return_MoM)
    'market_price_target_average,_detached_single_family_homes',
    'Log_Price',
    
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
    'sales_volume_MoM',
    'sales_volume_YoY',
    'year',
    'month',
    'provincial_gdp_real,_seasonally_adjusted',
    'national_gdp_nominal',
    'provincial_gdp_nominal',
    'gdp_per_cap_real',
    'national_debt_billions',
    'provincial_debt',
    'ontario_net_interprovincial_migration_monthly',
    'ontario_net_international_migration_monthly',
    'ontario_net_non_permanent_residents',
    'housing_starts_sfh,_monthly',
    'under_construction_sfh,_monthly',
    'completions__sfh,_monthly',
    'median_income_per_household_in_toronto',
    'cpi___national,_all_products', 
    'cpi___national,_core',
    '2y_bond',
    '10y_bond', 
    '5_year_fixed_mortgage_rate'
]

# Drop the specified columns from the dataframe
df.drop(columns=columns_to_drop, inplace=True, axis=1)

#drop indices that have NaNs from lag, delta, and average features
df.drop(df.index[0:24], inplace=True)

df.to_csv("data/processed_data.csv", index=True)
