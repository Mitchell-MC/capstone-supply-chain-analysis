# 🗺️ STATE NAME FEATURE ENGINEERING - Add this cell to your notebook

# FIPS State Code Mapping
fips_state_mapping = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California", 8: "Colorado",
    9: "Connecticut", 10: "Delaware", 11: "District of Columbia", 12: "Florida", 
    13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa",
    20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland", 
    25: "Massachusetts", 26: "Michigan", 27: "Minnesota", 28: "Mississippi", 29: "Missouri",
    30: "Montana", 31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio",
    40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island", 45: "South Carolina",
    46: "South Dakota", 47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont",
    51: "Virginia", 53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming"
}

print("🗺️ Adding state name features...")

# Add state name columns
df['origin_state_name'] = df['dms_origst'].map(fips_state_mapping)
df['dest_state_name'] = df['dms_destst'].map(fips_state_mapping)

# Create corridor names
df['corridor_names'] = df['origin_state_name'] + ' → ' + df['dest_state_name']

# Add same state indicator
df['is_same_state'] = (df['dms_origst'] == df['dms_destst']).astype(int)
df['same_state_name'] = df['is_same_state'].map({0: 'Interstate', 1: 'Intrastate'})

# Add region features
region_mapping = {
    'Alabama': 'Southeast', 'Alaska': 'West', 'Arizona': 'West', 'Arkansas': 'Southeast',
    'California': 'West', 'Colorado': 'West', 'Connecticut': 'Northeast', 'Delaware': 'Northeast',
    'District of Columbia': 'Northeast', 'Florida': 'Southeast', 'Georgia': 'Southeast',
    'Hawaii': 'West', 'Idaho': 'West', 'Illinois': 'Midwest', 'Indiana': 'Midwest',
    'Iowa': 'Midwest', 'Kansas': 'Midwest', 'Kentucky': 'Southeast', 'Louisiana': 'Southeast',
    'Maine': 'Northeast', 'Maryland': 'Northeast', 'Massachusetts': 'Northeast',
    'Michigan': 'Midwest', 'Minnesota': 'Midwest', 'Mississippi': 'Southeast',
    'Missouri': 'Midwest', 'Montana': 'West', 'Nebraska': 'Midwest', 'Nevada': 'West',
    'New Hampshire': 'Northeast', 'New Jersey': 'Northeast', 'New Mexico': 'West',
    'New York': 'Northeast', 'North Carolina': 'Southeast', 'North Dakota': 'Midwest',
    'Ohio': 'Midwest', 'Oklahoma': 'Southeast', 'Oregon': 'West', 'Pennsylvania': 'Northeast',
    'Rhode Island': 'Northeast', 'South Carolina': 'Southeast', 'South Dakota': 'Midwest',
    'Tennessee': 'Southeast', 'Texas': 'Southeast', 'Utah': 'West', 'Vermont': 'Northeast',
    'Virginia': 'Southeast', 'Washington': 'West', 'West Virginia': 'Southeast',
    'Wisconsin': 'Midwest', 'Wyoming': 'West'
}

df['origin_region'] = df['origin_state_name'].map(region_mapping)
df['dest_region'] = df['dest_state_name'].map(region_mapping)
df['corridor_region'] = df['origin_region'] + ' → ' + df['dest_region']

# Add distance band labels
dist_band_mapping = {
    1: '0-100 miles', 2: '100-250 miles', 3: '250-500 miles',
    4: '500-750 miles', 5: '750-1000 miles', 6: '1000-1500 miles',
    7: '1500-2000 miles', 8: '>2000 miles'
}
df['dist_band_label'] = df['dist_band'].map(dist_band_mapping)

# Add route type classification
df['route_type'] = np.where(df['dist_band'] <= 2, 'Nearshore',
                   np.where(df['dist_band'] >= 5, 'Long-Haul', 'Medium-Haul'))

print(f"✅ State name features added successfully!")
print(f"📊 New columns: origin_state_name, dest_state_name, corridor_names, is_same_state, same_state_name, origin_region, dest_region, corridor_region, dist_band_label, route_type")

# Display sample results
print(f"\n📋 Sample corridors:")
print(df['corridor_names'].head().tolist())
print(f"\n📈 Route type distribution:")
print(df['route_type'].value_counts()) 