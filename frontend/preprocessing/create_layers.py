import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

#gdf = gpd.read_file("/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/toronto158neighbourhoods.geojson")
#df_financials = pd.read_csv("/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/combined_neighbourhoods_financials.csv") 

df_master = pd.read_csv("/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/combined_neighbourhoods_financials.csv") 

df_master['geometry'] = gpd.GeoSeries.from_wkt(df_master['geometry'])

gdf_master = gpd.GeoDataFrame(df_master, geometry='geometry')

layer3_trreb = gdf_master.dissolve(by='TRREB_Zone').reset_index()
print(layer3_trreb.head())
#layer3_trreb.plot(edgecolor="black", color="lightblue")
#plt.savefig("layer3.png", dpi=300, bbox_inches="tight")
layer3_trreb.to_file("/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/layer3_trreb_zones.geojson", driver="GeoJSON")

# 3. LAYER 2: Macro Regions (West, Central, East)
def get_macro_region(zone):
    if str(zone).startswith('W'): return 'Toronto West'
    if str(zone).startswith('C'): return 'Toronto Central'
    if str(zone).startswith('E'): return 'Toronto East'
    return 'Unknown'

gdf_master['Macro_Region'] = gdf_master['TRREB_Zone'].apply(get_macro_region)

layer2_regions = gdf_master.dissolve(by='Macro_Region').reset_index()

TWcols = ['Median Price March 2025', 'Projected Price March 2050', 'Three Bedroom Avg. Lease Rate']
TWvals = [1365000, 5383252, 3691]
layer2_regions.loc[layer2_regions['Macro_Region'] == 'Toronto West', TWcols] = TWvals

TEcols = ['Median Price March 2025', 'Projected Price March 2050', 'Three Bedroom Avg. Lease Rate']
TEvals = [1175000, 4633935, 3009]
layer2_regions.loc[layer2_regions['Macro_Region'] == 'Toronto East', TEcols] = TEvals

TCcols = ['Median Price March 2025', 'Projected Price March 2050', 'Three Bedroom Avg. Lease Rate']
TCvals = [2175000, 8577709, 4410]
layer2_regions.loc[layer2_regions['Macro_Region'] == 'Toronto Central', TCcols] = TCvals

print(layer2_regions.head())
#layer2_regions.plot(edgecolor="black", color="lightblue")
#plt.savefig("layer2.png", dpi=300, bbox_inches="tight")
layer2_regions.to_file("/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/layer2_macro_regions.geojson", driver="GeoJSON")

gdf_master['City'] = 'Toronto'
layer1_city = gdf_master.dissolve(by='City').reset_index()

layer1cols = ['Median Price March 2025', 'Projected Price March 2050', 'Three Bedroom Avg. Lease Rate']
layer1vals = [1440000, 5679035, 3856]
layer1_city.loc[layer1_city['City'] == 'Toronto', layer1cols] = layer1vals

layer1_city = layer1_city.drop('Macro_Region', axis=1)  
print(layer1_city.head())
#layer1_city.plot(edgecolor="black", color="lightblue")
##plt.savefig("layer1.png", dpi=300, bbox_inches="tight")
layer1_city.to_file("/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/frontend/layer1_all_toronto.geojson", driver="GeoJSON")

print("All 3 map layers exported successfully!")