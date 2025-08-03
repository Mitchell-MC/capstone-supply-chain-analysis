import pandas as pd
import numpy as np

def check_foreign_region_codes():
    """
    Check all foreign region codes present in the FAF5 dataset
    """
    print("🔍 FOREIGN REGION CODES ANALYSIS")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv('FAF5.7_State.csv')
    
    # Check unique values in fr_orig and fr_dest
    print("📋 UNIQUE FOREIGN REGION CODES:")
    print(f"   fr_orig unique values: {sorted(df['fr_orig'].unique())}")
    print(f"   fr_dest unique values: {sorted(df['fr_dest'].unique())}")
    
    # Check for any codes beyond 801-808
    all_codes = set(df['fr_orig'].unique()) | set(df['fr_dest'].unique())
    print(f"\n🌍 ALL UNIQUE CODES FOUND: {sorted(all_codes)}")
    
    # Check for codes outside the 801-808 range
    expected_codes = set(range(801, 809))  # 801-808
    unexpected_codes = all_codes - expected_codes
    missing_codes = expected_codes - all_codes
    
    print(f"\n🔍 CODE ANALYSIS:")
    print(f"   Expected codes (801-808): {sorted(expected_codes)}")
    print(f"   Unexpected codes found: {sorted(unexpected_codes) if unexpected_codes else 'None'}")
    print(f"   Missing expected codes: {sorted(missing_codes) if missing_codes else 'None'}")
    
    # Count occurrences of each code
    print(f"\n📊 CODE FREQUENCY ANALYSIS:")
    print("   fr_orig frequency:")
    orig_counts = df['fr_orig'].value_counts().sort_index()
    for code, count in orig_counts.items():
        print(f"     Code {code}: {count:,} records")
    
    print(f"\n   fr_dest frequency:")
    dest_counts = df['fr_dest'].value_counts().sort_index()
    for code, count in dest_counts.items():
        print(f"     Code {code}: {count:,} records")
    
    # Check if there are any domestic codes (like 1 for US)
    domestic_codes = [code for code in all_codes if code < 800]
    if domestic_codes:
        print(f"\n🇺🇸 DOMESTIC CODES FOUND: {sorted(domestic_codes)}")
        print("   These likely represent US states/regions")
    
    # Provide updated mapping based on your codes
    print(f"\n🗺️ UPDATED FOREIGN REGION MAPPING:")
    updated_mapping = {
        801: 'Canada',
        802: 'Mexico', 
        803: 'Rest of Americas',
        804: 'Europe',
        805: 'Africa',
        806: 'Southwestern and Central Asia',
        807: 'Eastern Asia',
        808: 'Southeastern Asia and Oceania'
    }
    
    for code, name in updated_mapping.items():
        if code in all_codes:
            print(f"   {code}: {name} ✓")
        else:
            print(f"   {code}: {name} ✗ (not found in dataset)")
    
    # Check for any additional codes that need mapping
    additional_codes = [code for code in all_codes if code not in updated_mapping and code >= 800]
    if additional_codes:
        print(f"\n❓ ADDITIONAL CODES NEEDING MAPPING: {sorted(additional_codes)}")
        print("   These codes are present in the dataset but not in your mapping")
    
    return all_codes, updated_mapping

def analyze_code_distribution():
    """
    Analyze the distribution of foreign region codes
    """
    print(f"\n📊 CODE DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv('FAF5.7_State.csv')
    
    # Updated mapping
    foreign_region_mapping = {
        801: 'Canada',
        802: 'Mexico', 
        803: 'Rest of Americas',
        804: 'Europe',
        805: 'Africa',
        806: 'Southwestern and Central Asia',
        807: 'Eastern Asia',
        808: 'Southeastern Asia and Oceania'
    }
    
    # Add region names
    df['origin_foreign_region'] = df['fr_orig'].map(foreign_region_mapping)
    df['dest_foreign_region'] = df['fr_dest'].map(foreign_region_mapping)
    
    # Analyze by origin region
    print("🌍 ORIGIN REGION DISTRIBUTION:")
    origin_analysis = df.groupby('origin_foreign_region').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum'
    }).sort_values('tons_2023', ascending=False)
    
    for region, data in origin_analysis.iterrows():
        if pd.notna(region):
            volume = data['tons_2023']
            value = data['value_2023']
            value_density = value / volume if volume > 0 else 0
            print(f"   {region}: {volume:,.0f} tons, ${value:,.0f}, ${value_density:,.2f}/ton")
    
    # Analyze by destination region
    print(f"\n🌍 DESTINATION REGION DISTRIBUTION:")
    dest_analysis = df.groupby('dest_foreign_region').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum'
    }).sort_values('tons_2023', ascending=False)
    
    for region, data in dest_analysis.iterrows():
        if pd.notna(region):
            volume = data['tons_2023']
            value = data['value_2023']
            value_density = value / volume if volume > 0 else 0
            print(f"   {region}: {volume:,.0f} tons, ${value:,.0f}, ${value_density:,.2f}/ton")

def main():
    """
    Main function to check foreign region codes
    """
    print("🔍 FAF5 FOREIGN REGION CODES CHECK")
    print("=" * 60)
    
    # Check all codes present
    all_codes, mapping = check_foreign_region_codes()
    
    # Analyze distribution
    analyze_code_distribution()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"\n📋 SUMMARY:")
    print(f"   • Found {len(all_codes)} unique foreign region codes")
    print(f"   • Your mapping covers codes: {list(mapping.keys())}")
    print(f"   • Additional codes may need mapping if found")

if __name__ == "__main__":
    main() 