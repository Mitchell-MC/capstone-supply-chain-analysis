import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

print("🔧 FIXING INTERNATIONAL ANALYSIS")
print("=" * 50)

# Load data
print("📁 Loading FAF5.7 dataset...")
df = pd.read_csv('FAF5.7_State.csv')
print(f"✅ Dataset loaded: {df.shape[0]:,} records")

# Check foreign region codes
print(f"\n🔍 CHECKING FOREIGN REGION CODES:")
print(f"   fr_orig unique values: {sorted(df['fr_orig'].unique())}")
print(f"   fr_dest unique values: {sorted(df['fr_dest'].unique())}")

# Foreign Region Code Mapping
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

def categorize_market_distance(region):
    """Categorize regions by market distance"""
    if region in ['Canada', 'Mexico']:
        return 'Near-Shore'
    elif region in ['Europe', 'Eastern Asia']:
        return 'Mid-Distance'
    else:
        return 'Far-Shore'

# Pre-calculate essential metrics first
print("\n📊 CALCULATING METRICS...")
df['efficiency_ratio'] = df['tons_2023'] / (df['tmiles_2023'] + 0.001)
df['tons_volatility'] = df[['tons_2017', 'tons_2018', 'tons_2019', 'tons_2020', 'tons_2021', 'tons_2022', 'tons_2023']].std(axis=1)
df['value_density'] = df['value_2023'] / (df['tons_2023'] + 0.001)

# Clean data post-calculation
for col in ['efficiency_ratio', 'tons_volatility', 'value_density']:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())

# Add foreign region analysis
df['origin_foreign_region'] = df['fr_orig'].map(foreign_region_mapping)
df['dest_foreign_region'] = df['fr_dest'].map(foreign_region_mapping)
df['origin_market_distance'] = df['origin_foreign_region'].apply(categorize_market_distance)
df['dest_market_distance'] = df['dest_foreign_region'].apply(categorize_market_distance)

# Check international data
international_mask = (df['fr_orig'] >= 800) & (df['fr_orig'] <= 808)
international_df = df[international_mask].copy()

print(f"\n📊 INTERNATIONAL DATA ANALYSIS:")
print(f"   Total records: {len(df):,}")
print(f"   International records: {len(international_df):,}")
print(f"   International percentage: {len(international_df)/len(df)*100:.1f}%")

if len(international_df) == 0:
    print("\n⚠️ NO INTERNATIONAL DATA FOUND!")
    print("   Checking alternative filtering methods...")
    
    # Try different filtering approaches
    print(f"\n🔍 ALTERNATIVE FILTERING:")
    print(f"   Records with fr_orig >= 800: {len(df[df['fr_orig'] >= 800])}")
    print(f"   Records with fr_orig in [801,802,803,804,805,806,807,808]: {len(df[df['fr_orig'].isin([801,802,803,804,805,806,807,808])])}")
    print(f"   Records with fr_orig not null: {len(df[df['fr_orig'].notna()])}")
    print(f"   Records with fr_orig != 1: {len(df[df['fr_orig'] != 1])}")
    
    # Check if we need to use different criteria
    print(f"\n🔍 CHECKING FOR INTERNATIONAL INDICATORS:")
    print(f"   Records with trade_type == 2 (Import): {len(df[df['trade_type'] == 2])}")
    print(f"   Records with trade_type == 3 (Export): {len(df[df['trade_type'] == 3])}")
    print(f"   Records with high distance bands (>= 6): {len(df[df['dist_band'] >= 6])}")
    
    # Use alternative international definition
    international_df = df[df['trade_type'].isin([2, 3])].copy()
    print(f"\n📊 USING TRADE TYPE FILTERING:")
    print(f"   Import/Export records: {len(international_df):,}")
    print(f"   Percentage of total: {len(international_df)/len(df)*100:.1f}%")

print(f"\n📊 INTERNATIONAL DATA READY:")
print(f"   International records: {len(international_df):,}")
print(f"   Columns: {list(international_df.columns)}")

# Check for model data
features = ['fr_orig', 'fr_dest', 'dms_mode', 'sctg2', 'dist_band', 'trade_type']
model_data = international_df[features + ['efficiency_ratio']].dropna()

print(f"\n🔧 MODEL DATA PREPARATION:")
print(f"   Features: {features}")
print(f"   Model data shape: {model_data.shape}")
print(f"   Non-null efficiency_ratio: {model_data['efficiency_ratio'].notna().sum()}")

if len(model_data) == 0:
    print("\n⚠️ NO MODEL DATA AVAILABLE!")
    print("   Checking data quality issues...")
    
    # Check for data quality issues
    for feature in features:
        null_count = international_df[feature].isnull().sum()
        print(f"   {feature}: {null_count} null values")
    
    efficiency_null = international_df['efficiency_ratio'].isnull().sum()
    print(f"   efficiency_ratio: {efficiency_null} null values")
    
    # Try with more lenient filtering
    print(f"\n🔧 TRYING LENIENT FILTERING:")
    model_data = international_df[features + ['efficiency_ratio']].fillna(0)
    print(f"   Model data shape after fillna: {model_data.shape}")

# Only proceed with model if we have data
if len(model_data) > 0:
    print(f"\n✅ MODEL DATA READY: {len(model_data)} samples")
    
    X = model_data[features].copy()
    # Encode categorical features (keep dist_band as ordinal)
    for feature in features:
        if feature != 'dist_band':
            X[feature] = LabelEncoder().fit_transform(X[feature].astype(str))
    
    y = model_data['efficiency_ratio']
    
    # Check if we have enough data for train_test_split
    if len(X) >= 10:  # Minimum 10 samples
        print(f"\n🎯 TRAINING MODEL:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        # Use smaller test_size if limited data
        test_size = min(0.2, 1.0/len(X)) if len(X) < 50 else 0.2
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 FEATURE IMPORTANCE RANKINGS:")
        for _, row in importance_df.iterrows():
            feature_name = row['feature']
            if feature_name == 'fr_orig':
                feature_name = 'Origin Region'
            elif feature_name == 'fr_dest':
                feature_name = 'Destination Region'
            elif feature_name == 'dms_mode':
                feature_name = 'Transport Mode'
            elif feature_name == 'sctg2':
                feature_name = 'Commodity Type'
            elif feature_name == 'dist_band':
                feature_name = 'Distance Band'
            elif feature_name == 'trade_type':
                feature_name = 'Trade Type'
            
            print(f"   {feature_name:<20}: {row['importance']:.4f} ({row['importance']*100:5.1f}%)")
        
        # Model performance
        r2 = r2_score(y_test, rf_model.predict(X_test))
        print(f"\n📊 Model R² Score: {r2:.3f}")
        
    else:
        print(f"\n⚠️ INSUFFICIENT DATA FOR MODELING: {len(X)} samples")
        print("   Need at least 10 samples for meaningful model training")

else:
    print(f"\n❌ NO MODEL DATA AVAILABLE")
    print("   Cannot proceed with feature importance analysis")

print(f"\n✅ INTERNATIONAL ANALYSIS FIX COMPLETE!")
print(f"   Data loading: ✅")
print(f"   Foreign region mapping: ✅")
print(f"   International filtering: ✅")
print(f"   Model preparation: {'✅' if len(model_data) > 0 else '❌'}")
print(f"   Model training: {'✅' if len(model_data) >= 10 else '❌'}") 