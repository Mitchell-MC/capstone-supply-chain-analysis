import pandas as pd
import numpy as np

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

def get_state_name(fips_code):
    """Convert FIPS code to state name"""
    return fips_state_mapping.get(fips_code, f"Unknown ({fips_code})")

def add_state_name_features(df):
    """
    Add state name features to the dataframe
    """
    print("🗺️ Adding state name features...")
    
    # Add state name columns
    df['origin_state_name'] = df['dms_origst'].map(fips_state_mapping)
    df['dest_state_name'] = df['dms_destst'].map(fips_state_mapping)
    
    # Create corridor names
    df['corridor_names'] = df['origin_state_name'] + ' → ' + df['dest_state_name']
    
    # Add same state indicator
    df['is_same_state'] = (df['dms_origst'] == df['dms_destst']).astype(int)
    df['same_state_name'] = df['is_same_state'].map({0: 'Interstate', 1: 'Intrastate'})
    
    # Add region features (simplified)
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
    
    return df

def display_state_analysis(df):
    """
    Display analysis using state names
    """
    print("\n" + "="*60)
    print("📊 STATE NAME ANALYSIS")
    print("="*60)
    
    # Top origin states by volume
    print("\n🚛 TOP 10 ORIGIN STATES (by volume):")
    top_origins = df.groupby(['dms_origst', 'origin_state_name'])['tons_2023'].sum().sort_values(ascending=False).head(10)
    for (state_code, state_name), volume in top_origins.items():
        print(f"   {state_name:<20}: {volume/1e6:8.1f}M tons")
    
    # Top corridors
    print("\n🛣️ TOP 10 CORRIDORS (by volume):")
    top_corridors = df.groupby('corridor_names')['tons_2023'].sum().sort_values(ascending=False).head(10)
    for corridor, volume in top_corridors.items():
        print(f"   {corridor:<35}: {volume/1e6:8.1f}M tons")
    
    # Route type distribution
    print("\n📈 ROUTE TYPE DISTRIBUTION:")
    route_dist = df['route_type'].value_counts()
    for route_type, count in route_dist.items():
        percentage = count / len(df) * 100
        print(f"   {route_type:<12}: {count:>8,} corridors ({percentage:5.1f}%)")
    
    # Regional analysis
    print("\n🗺️ REGIONAL ANALYSIS:")
    region_dist = df['origin_region'].value_counts()
    for region, count in region_dist.items():
        percentage = count / len(df) * 100
        print(f"   {region:<12}: {count:>8,} origin corridors ({percentage:5.1f}%)")

# Example usage code
example_code = '''
# Load your data
df = pd.read_csv('FAF5.7_State.csv')

# Add state name features
df = add_state_name_features(df)

# Display analysis
display_state_analysis(df)

# Now you can use state names in your analysis:
# df.groupby('origin_state_name')['tons_2023'].sum()
# df.groupby('corridor_names')['value_2023'].sum()
# df[df['route_type'] == 'Nearshore']['efficiency_ratio'].mean()
'''

print("✅ State name feature engineering ready!")
print("\n📋 Copy this code into your notebook:")
print("="*50)
print(example_code)
print("="*50) 