import pandas as pd
import numpy as np

# Load the data
print("📁 Loading FAF5.7 dataset...")
df = pd.read_csv('FAF5.7_State.csv')
print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} columns")

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

# Add state name columns
df['origin_state_name'] = df['dms_origst'].map(fips_state_mapping)
df['dest_state_name'] = df['dms_destst'].map(fips_state_mapping)

print("\n🌍 OVERSEAS DATA ANALYSIS")
print("=" * 50)

# Check for any non-US state codes (values not in our mapping)
print("🔍 CHECKING FOR NON-US STATE CODES:")
non_us_origins = df[~df['dms_origst'].isin(fips_state_mapping.keys())]
non_us_destinations = df[~df['dms_destst'].isin(fips_state_mapping.keys())]

print(f"   Non-US origin codes found: {len(non_us_origins)}")
print(f"   Non-US destination codes found: {len(non_us_destinations)}")

if len(non_us_origins) > 0:
    print(f"   Unique non-US origin codes: {sorted(non_us_origins['dms_origst'].unique())}")
if len(non_us_destinations) > 0:
    print(f"   Unique non-US destination codes: {sorted(non_us_destinations['dms_destst'].unique())}")

# Check for any special values that might indicate overseas
print(f"\n🔍 CHECKING FOR SPECIAL VALUES:")
print(f"   fr_orig unique values: {sorted(df['fr_orig'].dropna().unique())[:10]}")
print(f"   fr_dest unique values: {sorted(df['fr_dest'].dropna().unique())[:10]}")

# Check trade_type for international indicators
print(f"\n🔍 TRADE TYPE ANALYSIS:")
trade_type_counts = df['trade_type'].value_counts()
print("   Trade type distribution:")
for trade_type, count in trade_type_counts.items():
    percentage = count / len(df) * 100
    print(f"     {trade_type}: {count:,} ({percentage:.1f}%)")

# Check for any corridor names that might indicate overseas
print(f"\n🔍 CHECKING CORRIDOR NAMES FOR OVERSEAS INDICATORS:")
# Add corridor names
df['corridor_names'] = df['origin_state_name'] + ' → ' + df['dest_state_name']

# Look for any corridor names that contain "Unknown" (indicating non-US codes)
unknown_corridors = df[df['origin_state_name'].str.contains('Unknown') | df['dest_state_name'].str.contains('Unknown')]
print(f"   Corridors with unknown state codes: {len(unknown_corridors)}")

if len(unknown_corridors) > 0:
    print("   Sample unknown corridors:")
    for corridor in unknown_corridors['corridor_names'].head(5):
        print(f"     {corridor}")

# Check for any high-value corridors that might be international
print(f"\n🔍 HIGH-VALUE CORRIDORS ANALYSIS:")
high_value = df[df['value_2023'] > df['value_2023'].quantile(0.99)]
print(f"   Top 1% value corridors: {len(high_value)}")
print("   Sample high-value corridors:")
for _, row in high_value.head(5).iterrows():
    print(f"     {row['corridor_names']}: ${row['value_2023']/1e3:.1f}K")

# Check for any unusual distance bands that might indicate overseas
print(f"\n🔍 DISTANCE BAND ANALYSIS:")
dist_band_counts = df['dist_band'].value_counts().sort_index()
print("   Distance band distribution:")
for band, count in dist_band_counts.items():
    percentage = count / len(df) * 100
    print(f"     Band {band}: {count:,} ({percentage:.1f}%)")

# Check if there are any records with very high distances
print(f"\n🔍 EXTREME DISTANCE ANALYSIS:")
max_distance_band = df['dist_band'].max()
print(f"   Maximum distance band: {max_distance_band}")
if max_distance_band > 8:
    print(f"   Records with distance band > 8: {len(df[df['dist_band'] > 8])}")

print(f"\n✅ OVERSEAS DATA ANALYSIS COMPLETE!")
print(f"📊 Dataset appears to be {'domestic US only' if len(non_us_origins) == 0 and len(non_us_destinations) == 0 else 'contains international data'}") 