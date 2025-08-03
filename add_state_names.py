import json
import re

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

# Code to add state name columns
state_name_code = '''
# Add state name columns
df['origin_state_name'] = df['dms_origst'].map(fips_state_mapping)
df['dest_state_name'] = df['dms_destst'].map(fips_state_mapping)
df['corridor_names'] = df['origin_state_name'] + ' → ' + df['dest_state_name']

print(f"🗺️ State names added successfully")
print(f"📊 Sample corridors: {df['corridor_names'].head().tolist()}")
'''

print("✅ State name mapping code ready!")
print("📋 Add this code after loading your data:")
print("\n" + "="*50)
print("FIPS STATE MAPPING:")
print("="*50)
for code, name in fips_state_mapping.items():
    print(f"{code:2d}: {name}")
print("="*50)
print("\nCODE TO ADD:")
print(state_name_code) 