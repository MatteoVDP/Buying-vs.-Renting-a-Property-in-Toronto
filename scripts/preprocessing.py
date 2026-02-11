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

df['Log_Price'] = np.log(df['market_price_target_average,_detached_single_family_homes'])

df['GDP_Growth_YoY'] = df['national_gdp_real,_seasonally_adjusted'].pct_change(12)
df['National_Pop_Growth_YoY'] = df['national_pop'].pct_change(12)
df['Provincial_Pop_Growth_YoY'] = df['provincial_pop'].pct_change(12)
df['Municipal_Pop_Growth_YoY'] = df['municipal_pop'].pct_change(12)
df['Inflation_Rate_YoY'] = df['cpi___national,_all_products'].pct_change(12)
df['Labour_Force_Growth_YoY'] = df['total_labour_force_size_people_15_or_over,_employed_or_actively_seeking_work'].pct_change(12)
df['Inflation_Rate'] = df['cpi___national,_core'].pct_change(12)

df['Real_Income'] = pd.to_numeric(df['median_income_per_household_in_toronto']) / df['cpi___national,_all_products']
df['Migration_Rate'] = (df['ontario_net_interprovincial_migration_monthly'] + df['ontario_net_international_migration_monthly'])/df['provincial_pop']
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

df.drop(df.index[0:12], inplace=True)

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

df.to_clipboard()
