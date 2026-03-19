import pandas as pd

gdf_neighbourhoods = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/gdf_neighbourhoods.csv')
trreb_data = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/TRREB-data-by-subdivision.csv')

# 2. Clean the 'Area' column in the TRREB data to match the 'TRREB_Zone' column
# This changes "Toronto W01" -> "W01"
trreb_data['TRREB_Zone'] = trreb_data['Area'].str.replace('Toronto ', '', regex=False)

# 3. Merge the datasets
# We use a 'left' join to ensure we keep all 158 original neighbourhood rows,
# attaching the financial data wherever the TRREB_Zone matches.
merged_df = gdf_neighbourhoods.merge(trreb_data, on='TRREB_Zone', how='left')
merged_df = merged_df.drop(columns=['Area', 'PARENT_AREA_ID', 'AREA_ATTR_ID', 'AREA_ID', 'Unnamed: 0'])  # Drop the original 'Area' column from TRREB data, since we have 'TRREB_Zone' now
# 4. Save the combined dataset to a new CSV file
merged_df.to_file('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/combined_neighbourhoods_financials.geojson', driver="GeoJSON")