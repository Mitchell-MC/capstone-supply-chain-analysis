# ============================================================================
# OPTIMIZED EXPERT MODE RISK CALCULATION
# ============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Tuple

def calculate_expert_mode_risk_optimized(df: pd.DataFrame, mode_risk_scores: Dict[int, int]) -> pd.DataFrame:
    """
    Optimized version of calculate_expert_mode_risk function.
    
    This version uses vectorized operations and efficient pandas methods
    to significantly reduce processing time.
    """
    
    print("🚀 Starting optimized expert mode risk calculation...")
    
    # Create a copy to avoid modifying original
    df_optimized = df.copy()
    
    # Vectorized risk score assignment
    df_optimized['transport_risk_score'] = df_optimized['dms_mode'].map(mode_risk_scores).fillna(5)
    
    # Calculate risk-weighted value using vectorized operations
    df_optimized['risk_weighted_value'] = (
        df_optimized['value_2023'] * 
        (1 - (df_optimized['transport_risk_score'] - 1) / 9)  # Normalize risk score to 0-1
    )
    
    # Calculate risk-adjusted resilience if resilience_score exists
    if 'resilience_score' in df_optimized.columns:
        df_optimized['risk_adjusted_resilience'] = (
            df_optimized['resilience_score'] * 
            (1 - (df_optimized['transport_risk_score'] - 1) / 9)
        )
    
    # Use groupby operations for efficient aggregation
    risk_analysis = df_optimized.groupby('dms_mode').agg({
        'transport_risk_score': ['mean', 'std'],
        'value_2023': ['sum', 'mean'],
        'risk_weighted_value': ['sum', 'mean'],
        'tons_2023': ['sum', 'mean']
    }).round(2)
    
    # Flatten column names
    risk_analysis.columns = ['_'.join(col).strip() for col in risk_analysis.columns]
    
    # Calculate risk metrics efficiently
    total_value = df_optimized['value_2023'].sum()
    total_risk_weighted_value = df_optimized['risk_weighted_value'].sum()
    
    # Identify high-risk modes (risk score >= 7)
    high_risk_modes = df_optimized[df_optimized['transport_risk_score'] >= 7]
    high_risk_value = high_risk_modes['value_2023'].sum()
    
    print(f"✅ Optimized expert mode risk calculation completed!")
    print(f"   Total value: ${total_value:,.0f}")
    print(f"   Risk-weighted value: ${total_risk_weighted_value:,.0f}")
    print(f"   High-risk value: ${high_risk_value:,.0f}")
    
    return df_optimized

def replace_slow_function_in_notebook():
    """
    Replace the slow calculate_expert_mode_risk function with the optimized version.
    """
    
    import json
    
    # Read the notebook
    with open('Supply_Chain_Volatility_Intl.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and replace the slow function
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def calculate_expert_mode_risk' in source:
                # Replace with optimized version
                optimized_source = '''def calculate_expert_mode_risk(df, mode_risk_scores):
    """
    Optimized version of calculate_expert_mode_risk function.
    """
    print("🚀 Starting optimized expert mode risk calculation...")
    
    # Create a copy to avoid modifying original
    df_optimized = df.copy()
    
    # Vectorized risk score assignment
    df_optimized['transport_risk_score'] = df_optimized['dms_mode'].map(mode_risk_scores).fillna(5)
    
    # Calculate risk-weighted value using vectorized operations
    df_optimized['risk_weighted_value'] = (
        df_optimized['value_2023'] * 
        (1 - (df_optimized['transport_risk_score'] - 1) / 9)
    )
    
    # Calculate risk-adjusted resilience if resilience_score exists
    if 'resilience_score' in df_optimized.columns:
        df_optimized['risk_adjusted_resilience'] = (
            df_optimized['resilience_score'] * 
            (1 - (df_optimized['transport_risk_score'] - 1) / 9)
        )
    
    # Use groupby operations for efficient aggregation
    risk_analysis = df_optimized.groupby('dms_mode').agg({
        'transport_risk_score': ['mean', 'std'],
        'value_2023': ['sum', 'mean'],
        'risk_weighted_value': ['sum', 'mean'],
        'tons_2023': ['sum', 'mean']
    }).round(2)
    
    # Flatten column names
    risk_analysis.columns = ['_'.join(col).strip() for col in risk_analysis.columns]
    
    # Calculate risk metrics efficiently
    total_value = df_optimized['value_2023'].sum()
    total_risk_weighted_value = df_optimized['risk_weighted_value'].sum()
    
    # Identify high-risk modes (risk score >= 7)
    high_risk_modes = df_optimized[df_optimized['transport_risk_score'] >= 7]
    high_risk_value = high_risk_modes['value_2023'].sum()
    
    print(f"✅ Optimized expert mode risk calculation completed!")
    print(f"   Total value: ${total_value:,.0f}")
    print(f"   Risk-weighted value: ${total_risk_weighted_value:,.0f}")
    print(f"   High-risk value: ${high_risk_value:,.0f}")
    
    return df_optimized'''
                
                cell['source'] = optimized_source.split('\n')
                print("✅ Replaced slow function with optimized version")
                break
    
    # Write the updated notebook
    with open('Supply_Chain_Volatility_Intl.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("🎯 Notebook updated with optimized function!")

if __name__ == "__main__":
    replace_slow_function_in_notebook() 