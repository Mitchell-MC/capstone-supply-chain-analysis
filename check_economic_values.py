#!/usr/bin/env python3
"""
Check Economic Values in FAF5.7 Dataset
=======================================

This script analyzes the economic values in the dataset to understand
data quality issues and scaling problems.
"""

import pandas as pd
import numpy as np

def analyze_economic_values():
    """Analyze economic values in the dataset"""
    print("🔍 ANALYZING ECONOMIC VALUES IN FAF5.7 DATASET")
    print("=" * 60)
    
    # Load dataset
    try:
        df = pd.read_csv('FAF5.7_State.csv')
        print(f"✅ Dataset loaded: {df.shape[0]:,} records")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Basic value analysis
    print("\n📊 BASIC VALUE ANALYSIS:")
    print(f"   • Total value 2023: ${df['value_2023'].sum():,.0f}")
    print(f"   • Total tons 2023: {df['tons_2023'].sum():,.0f}")
    print(f"   • Zero value records: {len(df[df['value_2023'] == 0]):,}")
    print(f"   • Non-zero value records: {len(df[df['value_2023'] > 0]):,}")
    
    # Value statistics
    print("\n📈 VALUE STATISTICS:")
    value_stats = df['value_2023'].describe()
    print(f"   • Mean: ${value_stats['mean']:.2f}")
    print(f"   • Median: ${value_stats['50%']:.2f}")
    print(f"   • Max: ${value_stats['max']:.2f}")
    print(f"   • 75th percentile: ${value_stats['75%']:.2f}")
    
    # Sample of highest values
    print("\n🏆 TOP 10 HIGHEST VALUES:")
    top_values = df.nlargest(10, 'value_2023')[['value_2023', 'tons_2023', 'fr_orig', 'fr_dest']]
    for idx, row in top_values.iterrows():
        print(f"   ${row['value_2023']:,.2f} | {row['tons_2023']:,.2f} tons | Origin: {row['fr_orig']} | Dest: {row['fr_dest']}")
    
    # International analysis
    print("\n🌍 INTERNATIONAL ANALYSIS:")
    international = df[df['fr_orig'] >= 800]
    print(f"   • International records: {len(international):,}")
    print(f"   • International value sum: ${international['value_2023'].sum():,.0f}")
    print(f"   • International tons sum: {international['tons_2023'].sum():,.0f}")
    
    # International value statistics
    if len(international) > 0:
        intl_value_stats = international['value_2023'].describe()
        print(f"   • International mean value: ${intl_value_stats['mean']:.2f}")
        print(f"   • International max value: ${intl_value_stats['max']:.2f}")
        
        # Top international values
        print("\n🌍 TOP 10 INTERNATIONAL VALUES:")
        top_intl = international.nlargest(10, 'value_2023')[['value_2023', 'tons_2023', 'fr_orig', 'fr_dest']]
        for idx, row in top_intl.iterrows():
            print(f"   ${row['value_2023']:,.2f} | {row['tons_2023']:,.2f} tons | Origin: {row['fr_orig']} | Dest: {row['fr_dest']}")
    
    # Value density analysis
    print("\n💰 VALUE DENSITY ANALYSIS:")
    non_zero = df[df['value_2023'] > 0]
    non_zero['value_density'] = non_zero['value_2023'] / (non_zero['tons_2023'] + 0.001)
    
    density_stats = non_zero['value_density'].describe()
    print(f"   • Mean value density: ${density_stats['mean']:.2f}/ton")
    print(f"   • Median value density: ${density_stats['50%']:.2f}/ton")
    print(f"   • Max value density: ${density_stats['max']:.2f}/ton")
    
    # Check for realistic economic scale
    print("\n🔍 ECONOMIC SCALE ASSESSMENT:")
    total_value_billions = df['value_2023'].sum() / 1e9
    print(f"   • Total value in billions: ${total_value_billions:.2f}B")
    
    if total_value_billions < 1:
        print("   ⚠️  Values appear to be in thousands, not millions/billions")
        print("   💡 Scaling factor of 1000x recommended for realistic analysis")
    elif total_value_billions < 100:
        print("   ⚠️  Values appear to be in millions, may need scaling")
    else:
        print("   ✅ Values appear to be in appropriate economic scale")
    
    # Sample of typical values
    print("\n📋 SAMPLE OF TYPICAL VALUES:")
    sample = df[df['value_2023'] > 0].sample(min(20, len(df)))[['value_2023', 'tons_2023']]
    for idx, row in sample.iterrows():
        print(f"   ${row['value_2023']:,.2f} | {row['tons_2023']:,.2f} tons")

if __name__ == "__main__":
    analyze_economic_values() 