import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """
    Load FAF5 data and prepare for international market analysis
    """
    print("📊 LOADING INTERNATIONAL FREIGHT DATASET...")
    
    # Load the dataset
    df = pd.read_csv('FAF5.7_State.csv')
    
    # Create detailed foreign region mapping
    foreign_region_mapping = {
        801: 'Canada',
        802: 'Mexico', 
        803: 'Europe',
        804: 'Asia',
        805: 'South America',
        806: 'Africa',
        807: 'Oceania',
        808: 'Other International'
    }
    
    # Add foreign region names
    df['origin_foreign_region'] = df['fr_orig'].map(foreign_region_mapping)
    df['dest_foreign_region'] = df['fr_dest'].map(foreign_region_mapping)
    
    # Create market distance categories
    def categorize_market_distance(region):
        if region in ['Canada', 'Mexico']:
            return 'Near-Shore'
        elif region in ['Europe', 'Asia']:
            return 'Mid-Distance'
        else:
            return 'Far-Shore'
    
    df['origin_market_distance'] = df['origin_foreign_region'].apply(categorize_market_distance)
    df['dest_market_distance'] = df['dest_foreign_region'].apply(categorize_market_distance)
    
    # Create trade type labels
    trade_type_mapping = {
        1: 'Domestic',
        2: 'Import',
        3: 'Export'
    }
    df['trade_type_label'] = df['trade_type'].map(trade_type_mapping)
    
    print(f"📊 Dataset loaded: {df.shape[0]:,} international freight records")
    print(f"🌍 Foreign region codes found: {sorted(df['fr_orig'].unique())}")
    
    return df

def analyze_market_distance_profiles(df):
    """
    Analyze freight patterns by market distance (near-shore vs far-shore)
    """
    print("\n🌍 INTERNATIONAL MARKET DISTANCE ANALYSIS")
    print("=" * 60)
    
    # Origin market analysis
    print(f"\n📤 ORIGIN MARKET DISTANCE PROFILES (2023):")
    origin_analysis = df.groupby('origin_market_distance').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count'
    }).sort_values('tons_2023', ascending=False)
    
    total_volume = df['tons_2023'].sum()
    total_value = df['value_2023'].sum()
    
    for distance, data in origin_analysis.iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        routes = data['trade_type_label']
        volume_pct = volume / total_volume * 100
        value_pct = value / total_value * 100
        value_density = value / volume if volume > 0 else 0
        
        print(f"\n   {distance} Markets:")
        print(f"     Volume: {volume:,.0f} tons ({volume_pct:.1f}%)")
        print(f"     Value: ${value:,.0f} ({value_pct:.1f}%)")
        print(f"     Routes: {routes:,}")
        print(f"     Value Density: ${value_density:,.2f}/ton")
    
    # Destination market analysis
    print(f"\n📥 DESTINATION MARKET DISTANCE PROFILES (2023):")
    dest_analysis = df.groupby('dest_market_distance').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count'
    }).sort_values('tons_2023', ascending=False)
    
    for distance, data in dest_analysis.iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        routes = data['trade_type_label']
        volume_pct = volume / total_volume * 100
        value_pct = value / total_value * 100
        value_density = value / volume if volume > 0 else 0
        
        print(f"\n   {distance} Markets:")
        print(f"     Volume: {volume:,.0f} tons ({volume_pct:.1f}%)")
        print(f"     Value: ${value:,.0f} ({value_pct:.1f}%)")
        print(f"     Routes: {routes:,}")
        print(f"     Value Density: ${value_density:,.2f}/ton")

def analyze_specific_region_profiles(df):
    """
    Analyze detailed profiles for each foreign region code
    """
    print(f"\n🌍 DETAILED REGION PROFILES BY CODE")
    print("=" * 60)
    
    # Origin region analysis
    print(f"\n📤 ORIGIN REGION PROFILES (2023):")
    origin_regions = df.groupby('origin_foreign_region').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count',
        'dms_mode': 'mean'
    }).sort_values('tons_2023', ascending=False)
    
    for region, data in origin_regions.iterrows():
        if pd.notna(region):
            volume = data['tons_2023']
            value = data['value_2023']
            routes = data['trade_type_label']
            avg_mode = data['dms_mode']
            value_density = value / volume if volume > 0 else 0
            
            print(f"\n   {region} (Code {get_region_code(region)}):")
            print(f"     Volume: {volume:,.0f} tons")
            print(f"     Value: ${value:,.0f}")
            print(f"     Routes: {routes:,}")
            print(f"     Value Density: ${value_density:,.2f}/ton")
            print(f"     Avg Transport Mode: {avg_mode:.1f}")
    
    # Destination region analysis
    print(f"\n📥 DESTINATION REGION PROFILES (2023):")
    dest_regions = df.groupby('dest_foreign_region').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count',
        'dms_mode': 'mean'
    }).sort_values('tons_2023', ascending=False)
    
    for region, data in dest_regions.iterrows():
        if pd.notna(region):
            volume = data['tons_2023']
            value = data['value_2023']
            routes = data['trade_type_label']
            avg_mode = data['dms_mode']
            value_density = value / volume if volume > 0 else 0
            
            print(f"\n   {region} (Code {get_region_code(region)}):")
            print(f"     Volume: {volume:,.0f} tons")
            print(f"     Value: ${value:,.0f}")
            print(f"     Routes: {routes:,}")
            print(f"     Value Density: ${value_density:,.2f}/ton")
            print(f"     Avg Transport Mode: {avg_mode:.1f}")

def get_region_code(region_name):
    """Helper function to get region code from name"""
    region_mapping = {
        'Canada': 801,
        'Mexico': 802,
        'Europe': 803,
        'Asia': 804,
        'South America': 805,
        'Africa': 806,
        'Oceania': 807,
        'Other International': 808
    }
    return region_mapping.get(region_name, 'Unknown')

def analyze_near_shore_vs_far_shore(df):
    """
    Detailed comparison of near-shore vs far-shore markets
    """
    print(f"\n🌍 NEAR-SHORE VS FAR-SHORE COMPARISON")
    print("=" * 60)
    
    # Near-shore markets (Canada, Mexico)
    near_shore_df = df[df['origin_foreign_region'].isin(['Canada', 'Mexico']) | 
                      df['dest_foreign_region'].isin(['Canada', 'Mexico'])]
    
    # Far-shore markets (Other International)
    far_shore_df = df[df['origin_foreign_region'].isin(['Europe', 'Asia', 'South America', 'Africa', 'Oceania']) | 
                      df['dest_foreign_region'].isin(['Europe', 'Asia', 'South America', 'Africa', 'Oceania'])]
    
    print(f"\n📊 MARKET COMPARISON (2023):")
    
    # Near-shore analysis
    near_volume = near_shore_df['tons_2023'].sum()
    near_value = near_shore_df['value_2023'].sum()
    near_routes = len(near_shore_df)
    near_value_density = near_value / near_volume if near_volume > 0 else 0
    
    print(f"\n   Near-Shore Markets (Canada/Mexico):")
    print(f"     Volume: {near_volume:,.0f} tons")
    print(f"     Value: ${near_value:,.0f}")
    print(f"     Routes: {near_routes:,}")
    print(f"     Value Density: ${near_value_density:,.2f}/ton")
    
    # Far-shore analysis
    far_volume = far_shore_df['tons_2023'].sum()
    far_value = far_shore_df['value_2023'].sum()
    far_routes = len(far_shore_df)
    far_value_density = far_value / far_volume if far_volume > 0 else 0
    
    print(f"\n   Far-Shore Markets (Other International):")
    print(f"     Volume: {far_volume:,.0f} tons")
    print(f"     Value: ${far_value:,.0f}")
    print(f"     Routes: {far_routes:,}")
    print(f"     Value Density: ${far_value_density:,.2f}/ton")
    
    # Comparison metrics
    print(f"\n   COMPARISON METRICS:")
    volume_ratio = near_volume / far_volume if far_volume > 0 else 0
    value_ratio = near_value / far_value if far_value > 0 else 0
    density_ratio = near_value_density / far_value_density if far_value_density > 0 else 0
    
    print(f"     Volume Ratio (Near/Far): {volume_ratio:.2f}x")
    print(f"     Value Ratio (Near/Far): {value_ratio:.2f}x")
    print(f"     Value Density Ratio (Near/Far): {density_ratio:.2f}x")

def analyze_transport_modes_by_distance(df):
    """
    Analyze transport mode preferences by market distance
    """
    print(f"\n🚢 TRANSPORT MODES BY MARKET DISTANCE")
    print("=" * 60)
    
    # Origin market distance transport analysis
    print(f"\n📤 TRANSPORT MODES BY ORIGIN MARKET DISTANCE:")
    origin_mode_analysis = df.groupby(['origin_market_distance', 'dms_mode']).agg({
        'tons_2023': 'sum',
        'value_2023': 'sum'
    }).sort_values('tons_2023', ascending=False)
    
    for (distance, mode), data in origin_mode_analysis.head(15).iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        value_density = value / volume if volume > 0 else 0
        
        print(f"   {distance} Markets - Mode {mode}: {volume:,.0f} tons, ${value:,.0f}, ${value_density:,.2f}/ton")
    
    # Destination market distance transport analysis
    print(f"\n📥 TRANSPORT MODES BY DESTINATION MARKET DISTANCE:")
    dest_mode_analysis = df.groupby(['dest_market_distance', 'dms_mode']).agg({
        'tons_2023': 'sum',
        'value_2023': 'sum'
    }).sort_values('tons_2023', ascending=False)
    
    for (distance, mode), data in dest_mode_analysis.head(15).iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        value_density = value / volume if volume > 0 else 0
        
        print(f"   {distance} Markets - Mode {mode}: {volume:,.0f} tons, ${value:,.0f}, ${value_density:,.2f}/ton")

def analyze_trade_types_by_distance(df):
    """
    Analyze trade types (import/export) by market distance
    """
    print(f"\n📦 TRADE TYPES BY MARKET DISTANCE")
    print("=" * 60)
    
    # Origin market distance trade analysis
    print(f"\n📤 TRADE TYPES BY ORIGIN MARKET DISTANCE:")
    origin_trade_analysis = df.groupby(['origin_market_distance', 'trade_type_label']).agg({
        'tons_2023': 'sum',
        'value_2023': 'sum'
    }).sort_values('tons_2023', ascending=False)
    
    for (distance, trade_type), data in origin_trade_analysis.iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        value_density = value / volume if volume > 0 else 0
        
        print(f"   {distance} Markets - {trade_type}: {volume:,.0f} tons, ${value:,.0f}, ${value_density:,.2f}/ton")
    
    # Destination market distance trade analysis
    print(f"\n📥 TRADE TYPES BY DESTINATION MARKET DISTANCE:")
    dest_trade_analysis = df.groupby(['dest_market_distance', 'trade_type_label']).agg({
        'tons_2023': 'sum',
        'value_2023': 'sum'
    }).sort_values('tons_2023', ascending=False)
    
    for (distance, trade_type), data in dest_trade_analysis.iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        value_density = value / volume if volume > 0 else 0
        
        print(f"   {distance} Markets - {trade_type}: {volume:,.0f} tons, ${value:,.0f}, ${value_density:,.2f}/ton")

def main():
    """
    Main function to analyze international markets by distance
    """
    print("🌍 INTERNATIONAL MARKET DISTANCE ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Run analyses
    analyze_market_distance_profiles(df)
    analyze_specific_region_profiles(df)
    analyze_near_shore_vs_far_shore(df)
    analyze_transport_modes_by_distance(df)
    analyze_trade_types_by_distance(df)
    
    print(f"\n✅ INTERNATIONAL MARKET ANALYSIS COMPLETE!")
    print(f"\n📋 KEY INSIGHTS:")
    print(f"   • Near-shore markets (Canada/Mexico) vs Far-shore markets (Other International)")
    print(f"   • Transport mode preferences by market distance")
    print(f"   • Trade type patterns (Import/Export) by market distance")
    print(f"   • Value density differences between near and far markets")
    print(f"   • Regional profiles based on foreign region codes 801-808")

if __name__ == "__main__":
    main() 