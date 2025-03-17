import pandas as pd
import re

# This file gets the objectives of brownfield data from 3 different files, deletes irrelevant columns, and merges the data to create a single file

df1 = pd.read_csv('partial_analysis.csv')
df2 = pd.read_csv('popWeightedSviAndSentiment.csv')
df3 = pd.read_csv('distanceLayers.csv')

# Irrelevant columns of proximity objectives file
indices_to_delete = [1,2,4,5,7,9,10]
df3.drop(df3.columns[indices_to_delete], axis=1, inplace=True)

# Merge the dataframes based on the registry_id column
merged_df1 = pd.merge(df1, df2, on='registryid')
merged_df2 = pd.merge(merged_df1, df3, on='registryid')

# Reorder the columns
merged_df = merged_df2[["registryid","x","y","county_fips","state_fips","state_nuclear_restrictions","state_electricity_price","state_net_electricity_imports",\
                        "state_nuclear_inclusive_policy","pop_weight_svi","traditionally_regulated_state_energy_market","state_construction_labor_5yr_mean_annual_wage","count_protected_lands",\
                            "count_hazardous_facilities_within_5mi","fault_line_hazard","landslide_hazard","max_pga_lt_0o3g","hundred_year_flood_zone","open_water_or_wetland","pop_weight_sentiment",\
                                "slope_lt_12pct","population_center_dist","ret_facility_dist","count_nuclear_rd_within_100mi","substation_dist","road_dist","streamflow_50kgpm_within_20mi"]]

merged_df['state_nuclear_restrictions'] = merged_df['state_nuclear_restrictions'].fillna('')
merged_df.drop_duplicates(subset=['registryid'], inplace=True)
merged_df.fillna(0, inplace=True)

print(merged_df)
# Filter rows where 'state_nuclear_restrictions' column does not contain '(6)'. 6 is moratorium, 10 is geographical moratorium in territories.
filtered_df = merged_df[~merged_df['state_nuclear_restrictions'].str.contains('\{6\}')]
filtered_df = filtered_df[~filtered_df['state_nuclear_restrictions'].str.contains('\{10\}')]

# make these miles from meters
columns_to_divide = ["substation_dist","road_dist","population_center_dist","ret_facility_dist"]
filtered_df[columns_to_divide] = filtered_df[columns_to_divide] / 1600

# make nuclear restrictions a count of state nuclear restrictions
def convert_str_to_list(column):
    for i in range(len(column.values)):
        element = column.values[i]
        if type(element) == str:
            match = re.match(r"\{[^{}]+\}", element)
            if match:
                column.values[i] = int((len(element)-1)/2)

convert_str_to_list(filtered_df["state_nuclear_restrictions"])
filtered_df = filtered_df.replace({True: 1, False: 0})
filtered_df = filtered_df.fillna(0)
filtered_df = filtered_df.replace('', 0)
print(filtered_df)
filtered_df.to_csv("merged_dataset.csv",index=False)