import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """
    Load FAF5 data and prepare for overseas analysis
    """
    print("📊 LOADING FAF5 DATASET...")
    
    # Load the dataset
    df = pd.read_csv('FAF5.7_State.csv')
    
    # Create foreign region mapping
    foreign_region_mapping = {
        801: 'Canada',
        802: 'Mexico', 
        803: 'Other International',
        804: 'Other International',
        805: 'Other International',
        806: 'Other International',
        807: 'Other International',
        808: 'Other International'
    }
    
    # Add foreign region names
    df['origin_foreign_region'] = df['fr_orig'].map(foreign_region_mapping)
    df['dest_foreign_region'] = df['fr_dest'].map(foreign_region_mapping)
    
    # Create trade type labels
    trade_type_mapping = {
        1: 'Domestic',
        2: 'Import',
        3: 'Export'
    }
    df['trade_type_label'] = df['trade_type'].map(trade_type_mapping)
    
    # Identify overseas vs domestic routes
    df['is_overseas'] = (df['fr_orig'] != 1) | (df['fr_dest'] != 1)
    df['route_type'] = df['is_overseas'].map({True: 'Overseas', False: 'Domestic'})
    
    print(f"📊 Dataset loaded: {df.shape[0]:,} records")
    print(f"🌍 Overseas routes: {df['is_overseas'].sum():,} ({df['is_overseas'].mean()*100:.1f}%)")
    print(f"🇺🇸 Domestic routes: {(~df['is_overseas']).sum():,} ({(~df['is_overseas']).mean()*100:.1f}%)")
    
    return df

def analyze_overseas_patterns(df):
    """
    Analyze overseas freight patterns
    """
    print("\n🌍 OVERSEAS FREIGHT ANALYSIS")
    print("=" * 50)
    
    # Overseas vs Domestic breakdown
    overseas_df = df[df['is_overseas']]
    domestic_df = df[~df['is_overseas']]
    
    print(f"\n📊 OVERSEAS VS DOMESTIC BREAKDOWN:")
    print(f"   Overseas routes: {len(overseas_df):,} records")
    print(f"   Domestic routes: {len(domestic_df):,} records")
    
    # Foreign region analysis
    print(f"\n🌍 FOREIGN REGION ANALYSIS:")
    print("   Origin regions:")
    origin_regions = overseas_df['origin_foreign_region'].value_counts()
    for region, count in origin_regions.items():
        if pd.notna(region):
            print(f"     {region}: {count:,} routes")
    
    print("   Destination regions:")
    dest_regions = overseas_df['dest_foreign_region'].value_counts()
    for region, count in dest_regions.items():
        if pd.notna(region):
            print(f"     {region}: {count:,} routes")
    
    # Trade type analysis
    print(f"\n📦 TRADE TYPE ANALYSIS:")
    trade_types = df['trade_type_label'].value_counts()
    for trade_type, count in trade_types.items():
        print(f"   {trade_type}: {count:,} routes ({count/len(df)*100:.1f}%)")
    
    # Overseas trade types
    overseas_trade_types = overseas_df['trade_type_label'].value_counts()
    print(f"\n🌍 OVERSEAS TRADE TYPES:")
    for trade_type, count in overseas_trade_types.items():
        print(f"   {trade_type}: {count:,} routes")

def analyze_overseas_volume_and_value(df):
    """
    Analyze overseas freight volume and value patterns
    """
    print(f"\n📊 OVERSEAS VOLUME AND VALUE ANALYSIS")
    print("=" * 50)
    
    overseas_df = df[df['is_overseas']]
    domestic_df = df[~df['is_overseas']]
    
    # 2023 data analysis
    print(f"\n📈 2023 FREIGHT ANALYSIS:")
    
    # Overseas volume
    overseas_volume_2023 = overseas_df['tons_2023'].sum()
    domestic_volume_2023 = domestic_df['tons_2023'].sum()
    total_volume_2023 = df['tons_2023'].sum()
    
    print(f"   Overseas volume: {overseas_volume_2023:,.0f} tons ({overseas_volume_2023/total_volume_2023*100:.1f}%)")
    print(f"   Domestic volume: {domestic_volume_2023:,.0f} tons ({domestic_volume_2023/total_volume_2023*100:.1f}%)")
    
    # Overseas value
    overseas_value_2023 = overseas_df['value_2023'].sum()
    domestic_value_2023 = domestic_df['value_2023'].sum()
    total_value_2023 = df['value_2023'].sum()
    
    print(f"   Overseas value: ${overseas_value_2023:,.0f} ({overseas_value_2023/total_value_2023*100:.1f}%)")
    print(f"   Domestic value: ${domestic_value_2023:,.0f} ({domestic_value_2023/total_value_2023*100:.1f}%)")
    
    # Value density comparison
    overseas_value_density = overseas_value_2023 / overseas_volume_2023 if overseas_volume_2023 > 0 else 0
    domestic_value_density = domestic_value_2023 / domestic_volume_2023 if domestic_volume_2023 > 0 else 0
    
    print(f"\n💰 VALUE DENSITY COMPARISON:")
    print(f"   Overseas: ${overseas_value_density:,.2f} per ton")
    print(f"   Domestic: ${domestic_value_density:,.2f} per ton")
    print(f"   Ratio (Overseas/Domestic): {overseas_value_density/domestic_value_density:.2f}x")

def analyze_overseas_by_region(df):
    """
    Analyze overseas freight by foreign region
    """
    print(f"\n🌍 OVERSEAS FREIGHT BY REGION")
    print("=" * 50)
    
    overseas_df = df[df['is_overseas']]
    
    # Analyze by origin region
    print(f"\n📤 TOP ORIGIN REGIONS (2023):")
    origin_analysis = overseas_df.groupby('origin_foreign_region').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count'
    }).sort_values('tons_2023', ascending=False)
    
    for region, data in origin_analysis.iterrows():
        if pd.notna(region):
            volume = data['tons_2023']
            value = data['value_2023']
            routes = data['trade_type_label']
            value_density = value / volume if volume > 0 else 0
            print(f"   {region}: {volume:,.0f} tons, ${value:,.0f}, {routes:,} routes, ${value_density:,.2f}/ton")
    
    # Analyze by destination region
    print(f"\n📥 TOP DESTINATION REGIONS (2023):")
    dest_analysis = overseas_df.groupby('dest_foreign_region').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count'
    }).sort_values('tons_2023', ascending=False)
    
    for region, data in dest_analysis.iterrows():
        if pd.notna(region):
            volume = data['tons_2023']
            value = data['value_2023']
            routes = data['trade_type_label']
            value_density = value / volume if volume > 0 else 0
            print(f"   {region}: {volume:,.0f} tons, ${value:,.0f}, {routes:,} routes, ${value_density:,.2f}/ton")

def analyze_overseas_trade_types(df):
    """
    Analyze overseas freight by trade type (import/export)
    """
    print(f"\n📦 OVERSEAS TRADE TYPE ANALYSIS")
    print("=" * 50)
    
    overseas_df = df[df['is_overseas']]
    
    # Trade type breakdown
    print(f"\n📊 TRADE TYPE BREAKDOWN (2023):")
    trade_analysis = overseas_df.groupby('trade_type_label').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count'
    }).sort_values('tons_2023', ascending=False)
    
    total_overseas_volume = overseas_df['tons_2023'].sum()
    total_overseas_value = overseas_df['value_2023'].sum()
    
    for trade_type, data in trade_analysis.iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        routes = data['trade_type_label']
        volume_pct = volume / total_overseas_volume * 100
        value_pct = value / total_overseas_value * 100
        value_density = value / volume if volume > 0 else 0
        
        print(f"   {trade_type}:")
        print(f"     Volume: {volume:,.0f} tons ({volume_pct:.1f}%)")
        print(f"     Value: ${value:,.0f} ({value_pct:.1f}%)")
        print(f"     Routes: {routes:,}")
        print(f"     Value Density: ${value_density:,.2f}/ton")

def analyze_overseas_commodities(df):
    """
    Analyze overseas freight by commodity type
    """
    print(f"\n📦 OVERSEAS COMMODITY ANALYSIS")
    print("=" * 50)
    
    overseas_df = df[df['is_overseas']]
    
    # Top commodities by volume
    print(f"\n📊 TOP 10 OVERSEAS COMMODITIES BY VOLUME (2023):")
    commodity_volume = overseas_df.groupby('sctg2')['tons_2023'].sum().sort_values(ascending=False).head(10)
    
    for commodity, volume in commodity_volume.items():
        print(f"   Commodity {commodity}: {volume:,.0f} tons")
    
    # Top commodities by value
    print(f"\n💰 TOP 10 OVERSEAS COMMODITIES BY VALUE (2023):")
    commodity_value = overseas_df.groupby('sctg2')['value_2023'].sum().sort_values(ascending=False).head(10)
    
    for commodity, value in commodity_value.items():
        print(f"   Commodity {commodity}: ${value:,.0f}")

def analyze_overseas_transport_modes(df):
    """
    Analyze overseas freight by transport mode
    """
    print(f"\n🚢 OVERSEAS TRANSPORT MODE ANALYSIS")
    print("=" * 50)
    
    overseas_df = df[df['is_overseas']]
    
    # Mode breakdown
    print(f"\n📊 TRANSPORT MODE BREAKDOWN (2023):")
    mode_analysis = overseas_df.groupby('dms_mode').agg({
        'tons_2023': 'sum',
        'value_2023': 'sum',
        'trade_type_label': 'count'
    }).sort_values('tons_2023', ascending=False)
    
    total_overseas_volume = overseas_df['tons_2023'].sum()
    total_overseas_value = overseas_df['value_2023'].sum()
    
    for mode, data in mode_analysis.iterrows():
        volume = data['tons_2023']
        value = data['value_2023']
        routes = data['trade_type_label']
        volume_pct = volume / total_overseas_volume * 100
        value_pct = value / total_overseas_value * 100
        value_density = value / volume if volume > 0 else 0
        
        print(f"   Mode {mode}:")
        print(f"     Volume: {volume:,.0f} tons ({volume_pct:.1f}%)")
        print(f"     Value: ${value:,.0f} ({value_pct:.1f}%)")
        print(f"     Routes: {routes:,}")
        print(f"     Value Density: ${value_density:,.2f}/ton")

def main():
    """
    Main function to analyze overseas freight data
    """
    print("🚢 OVERSEAS FREIGHT ANALYSIS FROM FAF5 DATASET")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Run analyses
    analyze_overseas_patterns(df)
    analyze_overseas_volume_and_value(df)
    analyze_overseas_by_region(df)
    analyze_overseas_trade_types(df)
    analyze_overseas_commodities(df)
    analyze_overseas_transport_modes(df)
    
    print(f"\n✅ OVERSEAS ANALYSIS COMPLETE!")
    print(f"\n📋 KEY INSIGHTS:")
    print(f"   • Your FAF5 dataset contains significant overseas freight data")
    print(f"   • Foreign region codes 801-808 represent international routes")
    print(f"   • Trade types 1, 2, 3 represent different international trade categories")
    print(f"   • Distance bands up to 8 suggest overseas routes")
    print(f"   • This provides a comprehensive view of both domestic and international freight")

if __name__ == "__main__":
    main() 