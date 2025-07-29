#!/usr/bin/env python3
"""
Fix for efficiency_ratio quartile issue in FAF5 analysis
This script provides the corrected code to handle duplicate bin edges
"""

print("""
=== FIX FOR EFFICIENCY_RATIO QUARTILE ERROR ===

Replace this problematic line:
df['efficiency_quartile'] = pd.qcut(df['efficiency_ratio'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

With this corrected code:
""")

corrected_code = '''
# 1. EFFICIENCY QUARTILES (balanced by design) - FIXED VERSION
print("Creating efficiency quartiles with duplicate handling...")

# Examine the efficiency_ratio distribution
print(f"Efficiency ratio stats:")
print(f"  Min: {df['efficiency_ratio'].min():.6f}")
print(f"  Max: {df['efficiency_ratio'].max():.6f}")
print(f"  Zeros: {(df['efficiency_ratio'] == 0).sum():,}")
print(f"  Unique values: {df['efficiency_ratio'].nunique():,}")

# Method 1: Use duplicates='drop' parameter
try:
    df['efficiency_quartile'] = pd.qcut(
        df['efficiency_ratio'], 
        q=4, 
        labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
        duplicates='drop'  # Handle duplicate bin edges
    )
    print("✅ Method 1 (duplicates='drop') worked!")
    
except Exception as e:
    print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Create custom bins based on non-zero values
    print("Trying Method 2: Custom bins...")
    
    # Filter out zeros for quartile calculation
    non_zero_efficiency = df[df['efficiency_ratio'] > 0]['efficiency_ratio']
    
    if len(non_zero_efficiency) > 0:
        # Get quartile boundaries from non-zero values
        q25 = non_zero_efficiency.quantile(0.25)
        q50 = non_zero_efficiency.quantile(0.50)  
        q75 = non_zero_efficiency.quantile(0.75)
        
        print(f"Custom quartile boundaries: 0, {q25:.6f}, {q50:.6f}, {q75:.6f}, {non_zero_efficiency.max():.6f}")
        
        # Create categories with custom logic
        def assign_efficiency_quartile(value):
            if value == 0:
                return 'Low'  # Assign zeros to Low category
            elif value <= q25:
                return 'Low'
            elif value <= q50:
                return 'Medium-Low'
            elif value <= q75:
                return 'Medium-High'
            else:
                return 'High'
        
        df['efficiency_quartile'] = df['efficiency_ratio'].apply(assign_efficiency_quartile)
        print("✅ Method 2 (custom bins) worked!")
    
    else:
        print("❌ No non-zero efficiency values found!")
        # Fallback: Use simple high/low based on median
        median_val = df['efficiency_ratio'].median()
        df['efficiency_quartile'] = (df['efficiency_ratio'] > median_val).map({
            True: 'High', 
            False: 'Low'
        })
        print("✅ Fallback method: Simple High/Low classification")

# Verify the result
print(f"\\nEfficiency quartile distribution:")
print(df['efficiency_quartile'].value_counts())
print(f"Null values: {df['efficiency_quartile'].isnull().sum()}")
'''

print(corrected_code)

print("""
=== ADDITIONAL FIXES FOR OTHER QUARTILES ===

Also apply similar fixes to value_density_cat and growth_performance:
""")

additional_fixes = '''
# 2. VALUE DENSITY CATEGORIES (with duplicate handling)
value_per_ton = df['value_2023'] / (df['tons_2023'] + 0.001)
try:
    df['value_density_cat'] = pd.qcut(
        value_per_ton, 
        q=3, 
        labels=['Bulk', 'Mixed', 'High-Value'],
        duplicates='drop'
    )
except:
    # Fallback to manual bins if needed
    q33 = value_per_ton.quantile(0.33)
    q67 = value_per_ton.quantile(0.67)
    df['value_density_cat'] = pd.cut(
        value_per_ton,
        bins=[-np.inf, q33, q67, np.inf],
        labels=['Bulk', 'Mixed', 'High-Value']
    )

# 3. GROWTH PERFORMANCE (simple binary split)
median_growth = df['tons_growth_rate'].median()
df['growth_performance'] = (df['tons_growth_rate'] > median_growth).map({
    True: 'Growing', 
    False: 'Declining'
})
'''

print(additional_fixes)

print("""
=== HOW TO USE ===
1. Copy the corrected code above
2. Replace the problematic pd.qcut line in your notebook
3. Run the cell again
4. The quartiles should now work properly!
""")