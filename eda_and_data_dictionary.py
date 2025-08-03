#!/usr/bin/env python3
"""
Exploratory Data Analysis and Data Dictionary for FAF5.7 Dataset
================================================================

This script performs comprehensive EDA on the FAF5.7 freight dataset and
creates a detailed data dictionary documenting all variables, their meanings,
and data quality characteristics.

Author: Senior Data Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_dataset():
    """Load and perform initial exploration of the dataset"""
    print("🔍 EXPLORATORY DATA ANALYSIS - FAF5.7 DATASET")
    print("=" * 60)
    
    # Load dataset
    try:
        df = pd.read_csv('FAF5.7_State.csv')
        print(f"✅ Dataset loaded successfully: {df.shape[0]:,} records × {df.shape[1]} features")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    return df

def analyze_dataset_structure(df):
    """Analyze the overall structure of the dataset"""
    print("\n📊 DATASET STRUCTURE ANALYSIS")
    print("=" * 40)
    
    print(f"📈 Dataset Dimensions:")
    print(f"   • Rows: {df.shape[0]:,}")
    print(f"   • Columns: {df.shape[1]}")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types analysis
    print(f"\n📋 Data Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} columns")
    
    # Missing values analysis
    print(f"\n❓ Missing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    }).sort_values('Missing_Count', ascending=False)
    
    print(f"   • Columns with missing values: {len(missing_summary[missing_summary['Missing_Count'] > 0])}")
    print(f"   • Total missing values: {missing_data.sum():,}")
    
    # Show top missing columns
    if len(missing_summary[missing_summary['Missing_Count'] > 0]) > 0:
        print(f"\n   Top 10 columns with missing values:")
        for idx, row in missing_summary.head(10).iterrows():
            print(f"     • {row['Column']}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%)")
    
    return missing_summary

def analyze_numeric_columns(df):
    """Analyze numeric columns in detail"""
    print("\n🔢 NUMERIC COLUMNS ANALYSIS")
    print("=" * 35)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"📊 Found {len(numeric_cols)} numeric columns")
    
    numeric_analysis = {}
    
    for col in numeric_cols:
        print(f"\n📈 {col}:")
        stats = df[col].describe()
        
        # Basic statistics
        print(f"   • Mean: {stats['mean']:.2f}")
        print(f"   • Median: {stats['50%']:.2f}")
        print(f"   • Std Dev: {stats['std']:.2f}")
        print(f"   • Min: {stats['min']:.2f}")
        print(f"   • Max: {stats['max']:.2f}")
        print(f"   • Zero values: {(df[col] == 0).sum():,}")
        print(f"   • Null values: {df[col].isnull().sum():,}")
        
        # Store for data dictionary
        numeric_analysis[col] = {
            'mean': stats['mean'],
            'median': stats['50%'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'zero_count': (df[col] == 0).sum(),
            'null_count': df[col].isnull().sum(),
            'unique_values': df[col].nunique()
        }
    
    return numeric_analysis

def analyze_categorical_columns(df):
    """Analyze categorical columns in detail"""
    print("\n📝 CATEGORICAL COLUMNS ANALYSIS")
    print("=" * 40)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        categorical_cols = df.select_dtypes(include=['int64']).columns[:10]  # Use some int columns as categorical
    
    print(f"📊 Found {len(categorical_cols)} categorical columns")
    
    categorical_analysis = {}
    
    for col in categorical_cols:
        print(f"\n📋 {col}:")
        
        # Basic statistics
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        most_common = df[col].value_counts().head(5)
        
        print(f"   • Unique values: {unique_count:,}")
        print(f"   • Null values: {null_count:,}")
        print(f"   • Most common values:")
        for value, count in most_common.items():
            print(f"     - {value}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Store for data dictionary
        categorical_analysis[col] = {
            'unique_count': unique_count,
            'null_count': null_count,
            'most_common': most_common.to_dict()
        }
    
    return categorical_analysis

def analyze_time_series_columns(df):
    """Analyze time series columns (tons and value columns)"""
    print("\n📅 TIME SERIES ANALYSIS")
    print("=" * 25)
    
    # Find tons and value columns
    tons_cols = [col for col in df.columns if col.startswith('tons_')]
    value_cols = [col for col in df.columns if col.startswith('value_')]
    
    print(f"📊 Tons columns: {len(tons_cols)}")
    print(f"📊 Value columns: {len(value_cols)}")
    
    # Analyze tons columns
    if tons_cols:
        print(f"\n📈 Tons columns analysis:")
        for col in tons_cols:
            year = col.split('_')[1] if '_' in col else 'Unknown'
            total_tons = df[col].sum()
            print(f"   • {col} ({year}): {total_tons:,.0f} tons")
    
    # Analyze value columns
    if value_cols:
        print(f"\n💰 Value columns analysis:")
        for col in value_cols:
            year = col.split('_')[1] if '_' in col else 'Unknown'
            total_value = df[col].sum()
            print(f"   • {col} ({year}): ${total_value:,.0f}")
    
    return tons_cols, value_cols

def analyze_geographic_columns(df):
    """Analyze geographic and location columns"""
    print("\n🗺️ GEOGRAPHIC ANALYSIS")
    print("=" * 25)
    
    # Look for geographic columns
    geo_cols = [col for col in df.columns if any(term in col.lower() for term in ['orig', 'dest', 'state', 'region', 'fips'])]
    
    print(f"📊 Geographic columns found: {len(geo_cols)}")
    for col in geo_cols:
        print(f"   • {col}: {df[col].nunique()} unique values")
        if df[col].nunique() < 100:  # Show values if not too many
            unique_vals = df[col].dropna().unique()
            print(f"     Values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
    
    return geo_cols

def create_data_dictionary(df, numeric_analysis, categorical_analysis, tons_cols, value_cols, geo_cols):
    """Create comprehensive data dictionary"""
    print("\n📚 CREATING DATA DICTIONARY")
    print("=" * 35)
    
    # Define column categories and descriptions
    column_descriptions = {
        # Geographic identifiers
        'dms_origst': 'Origin state FIPS code (1-56)',
        'dms_destst': 'Destination state FIPS code (1-56)',
        'fr_orig': 'Foreign origin region code (801-808 for international)',
        'fr_dest': 'Foreign destination region code (801-808 for international)',
        
        # Transportation
        'dms_mode': 'Transport mode (1=Truck, 2=Rail, 3=Water, 4=Air, 5=Pipeline)',
        'tmiles_2023': 'Transport miles in 2023',
        
        # Commodity classification
        'sctg2': 'Standard Classification of Transported Goods (2-digit)',
        
        # Trade type
        'trade_type': 'Trade type (1=Domestic, 2=Import, 3=Export)',
        
        # Distance classification
        'dist_band': 'Distance band classification',
        
        # Tons data (time series)
        'tons_2017': 'Freight tons in 2017',
        'tons_2018': 'Freight tons in 2018',
        'tons_2019': 'Freight tons in 2019',
        'tons_2020': 'Freight tons in 2020',
        'tons_2021': 'Freight tons in 2021',
        'tons_2022': 'Freight tons in 2022',
        'tons_2023': 'Freight tons in 2023',
        'tons_2024': 'Projected freight tons in 2024',
        'tons_2030': 'Projected freight tons in 2030',
        'tons_2035': 'Projected freight tons in 2035',
        'tons_2040': 'Projected freight tons in 2040',
        'tons_2045': 'Projected freight tons in 2045',
        'tons_2050': 'Projected freight tons in 2050',
        
        # Value data (time series)
        'value_2017': 'Freight value in 2017 (thousands of dollars)',
        'value_2018': 'Freight value in 2018 (thousands of dollars)',
        'value_2019': 'Freight value in 2019 (thousands of dollars)',
        'value_2020': 'Freight value in 2020 (thousands of dollars)',
        'value_2021': 'Freight value in 2021 (thousands of dollars)',
        'value_2022': 'Freight value in 2022 (thousands of dollars)',
        'value_2023': 'Freight value in 2023 (thousands of dollars)',
        'value_2024': 'Projected freight value in 2024 (thousands of dollars)',
    }
    
    # Create data dictionary
    data_dict = []
    
    for col in df.columns:
        col_info = {
            'Column_Name': col,
            'Data_Type': str(df[col].dtype),
            'Description': column_descriptions.get(col, 'No description available'),
            'Missing_Count': df[col].isnull().sum(),
            'Missing_Percent': (df[col].isnull().sum() / len(df)) * 100,
            'Unique_Values': df[col].nunique()
        }
        
        # Add numeric-specific information
        if df[col].dtype in ['int64', 'float64']:
            if col in numeric_analysis:
                col_info.update({
                    'Mean': numeric_analysis[col]['mean'],
                    'Median': numeric_analysis[col]['median'],
                    'Std_Dev': numeric_analysis[col]['std'],
                    'Min': numeric_analysis[col]['min'],
                    'Max': numeric_analysis[col]['max'],
                    'Zero_Count': numeric_analysis[col]['zero_count']
                })
        
        # Add categorical-specific information
        if col in categorical_analysis:
            col_info['Most_Common_Values'] = str(categorical_analysis[col]['most_common'])
        
        # Categorize columns
        if col in tons_cols:
            col_info['Category'] = 'Tons_Data'
        elif col in value_cols:
            col_info['Category'] = 'Value_Data'
        elif col in geo_cols:
            col_info['Category'] = 'Geographic'
        elif 'mode' in col.lower():
            col_info['Category'] = 'Transportation'
        elif 'sctg' in col.lower():
            col_info['Category'] = 'Commodity'
        else:
            col_info['Category'] = 'Other'
        
        data_dict.append(col_info)
    
    return pd.DataFrame(data_dict)

def generate_quality_report(df):
    """Generate data quality report"""
    print("\n🔍 DATA QUALITY REPORT")
    print("=" * 25)
    
    # Overall quality metrics
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    print(f"📊 Overall Data Quality:")
    print(f"   • Data completeness: {completeness:.2f}%")
    print(f"   • Total missing cells: {missing_cells:,}")
    print(f"   • Records with any missing data: {df.isnull().any(axis=1).sum():,}")
    
    # Column quality assessment
    print(f"\n📋 Column Quality Assessment:")
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 0:
            print(f"   • {col}: {missing_pct:.1f}% missing")
    
    # Data consistency checks
    print(f"\n✅ Data Consistency Checks:")
    
    # Check for negative values in tons and value columns
    tons_cols = [col for col in df.columns if col.startswith('tons_')]
    value_cols = [col for col in df.columns if col.startswith('value_')]
    
    for col in tons_cols + value_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"   ⚠️  {col}: {negative_count} negative values")
        else:
            print(f"   ✅ {col}: No negative values")
    
    # Check for unrealistic values
    for col in tons_cols:
        max_val = df[col].max()
        if max_val > 1e6:
            print(f"   ⚠️  {col}: Maximum value {max_val:,.0f} may be unrealistic")
    
    for col in value_cols:
        max_val = df[col].max()
        if max_val > 1e6:
            print(f"   ⚠️  {col}: Maximum value {max_val:,.0f} may be unrealistic")

def save_data_dictionary(data_dict, filename='FAF5_Data_Dictionary.csv'):
    """Save data dictionary to CSV"""
    data_dict.to_csv(filename, index=False)
    print(f"\n💾 Data dictionary saved to: {filename}")
    
    # Also save as markdown for better readability
    markdown_filename = filename.replace('.csv', '.md')
    with open(markdown_filename, 'w') as f:
        f.write("# FAF5.7 Dataset Data Dictionary\n\n")
        f.write("This document provides a comprehensive description of all variables in the FAF5.7 freight dataset.\n\n")
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Records**: {len(data_dict):,}\n")
        f.write(f"- **Total Variables**: {len(data_dict):,}\n")
        f.write(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Variable Descriptions\n\n")
        f.write("| Column Name | Data Type | Description | Missing % | Category |\n")
        f.write("|-------------|-----------|-------------|-----------|----------|\n")
        
        for _, row in data_dict.iterrows():
            f.write(f"| {row['Column_Name']} | {row['Data_Type']} | {row['Description']} | {row['Missing_Percent']:.1f}% | {row['Category']} |\n")
    
    print(f"📝 Markdown documentation saved to: {markdown_filename}")

def main():
    """Main execution function"""
    print("🚛 FAF5.7 DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    df = load_and_explore_dataset()
    if df is None:
        return
    
    # Perform comprehensive analysis
    missing_summary = analyze_dataset_structure(df)
    numeric_analysis = analyze_numeric_columns(df)
    categorical_analysis = analyze_categorical_columns(df)
    tons_cols, value_cols = analyze_time_series_columns(df)
    geo_cols = analyze_geographic_columns(df)
    
    # Generate quality report
    generate_quality_report(df)
    
    # Create data dictionary
    data_dict = create_data_dictionary(df, numeric_analysis, categorical_analysis, tons_cols, value_cols, geo_cols)
    
    # Save results
    save_data_dictionary(data_dict)
    
    # Print summary
    print(f"\n✅ EDA COMPLETED SUCCESSFULLY!")
    print(f"📊 Dataset Summary:")
    print(f"   • Records: {df.shape[0]:,}")
    print(f"   • Variables: {df.shape[1]}")
    print(f"   • Tons columns: {len(tons_cols)}")
    print(f"   • Value columns: {len(value_cols)}")
    print(f"   • Geographic columns: {len(geo_cols)}")
    print(f"   • Data completeness: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1])) * 100:.1f}%")
    
    return df, data_dict

if __name__ == "__main__":
    df, data_dict = main() 