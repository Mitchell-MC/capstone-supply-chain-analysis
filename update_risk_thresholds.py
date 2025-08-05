import json
import pandas as pd
import numpy as np

def update_risk_thresholds():
    """
    Update risk thresholds to new ranges: >0.6, 0.3-0.6, <0.3
    """
    
    print("🔧 UPDATING RISK THRESHOLDS")
    print("=" * 50)
    
    # Read the existing notebook
    try:
        with open('Supply_Chain_Volatility_Intl.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        print("✅ Successfully loaded existing notebook")
    except Exception as e:
        print(f"❌ Error loading notebook: {e}")
        return False
    
    # Update the risk scoring cell with new thresholds
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('Risk Score Range:' in line for line in cell['source']):
            # Update the risk score reporting section
            updated_source = []
            for line in cell['source']:
                if 'High-Risk Corridors (>0.7):' in line:
                    updated_source.append('        print(f"   • High-Risk Corridors (>0.6): {len(risk_scores[risk_scores > 0.6]):,}")\n')
                elif 'Medium-Risk Corridors (0.4-0.7):' in line:
                    updated_source.append('        print(f"   • Medium-Risk Corridors (0.3-0.6): {len(risk_scores[(risk_scores > 0.3) & (risk_scores <= 0.6)]):,}")\n')
                elif 'Low-Risk Corridors (<0.4):' in line:
                    updated_source.append('        print(f"   • Low-Risk Corridors (<0.3): {len(risk_scores[risk_scores <= 0.3]):,}")\n')
                else:
                    updated_source.append(line)
            notebook['cells'][i]['source'] = updated_source
            break
    
    # Update the executive dashboard cell with new thresholds
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('high_risk_count = len(df[df[\'comprehensive_risk_score\'] > 0.7])' in line for line in cell['source']):
            # Update the executive dashboard recommendations
            updated_source = []
            for line in cell['source']:
                if 'high_risk_count = len(df[df[\'comprehensive_risk_score\'] > 0.7])' in line:
                    updated_source.append('            high_risk_count = len(df[df[\'comprehensive_risk_score\'] > 0.6])\n')
                elif 'medium_risk_count = len(df[(df[\'comprehensive_risk_score\'] > 0.4) & (df[\'comprehensive_risk_score\'] <= 0.7)])' in line:
                    updated_source.append('            medium_risk_count = len(df[(df[\'comprehensive_risk_score\'] > 0.3) & (df[\'comprehensive_risk_score\'] <= 0.6)])\n')
                else:
                    updated_source.append(line)
            notebook['cells'][i]['source'] = updated_source
            break
    
    # Save the modified notebook
    try:
        with open('Supply_Chain_Volatility_Intl.ipynb', 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("✅ Successfully updated risk thresholds in notebook")
        print("📁 Modified: Supply_Chain_Volatility_Intl.ipynb")
        return True
    except Exception as e:
        print(f"❌ Error saving notebook: {e}")
        return False

if __name__ == "__main__":
    success = update_risk_thresholds()
    if success:
        print("\n🎉 RISK THRESHOLDS UPDATED!")
        print("\n📋 New Thresholds:")
        print("• High-Risk: >0.6")
        print("• Medium-Risk: 0.3-0.6")
        print("• Low-Risk: <0.3")
    else:
        print("\n❌ Risk threshold update failed. Please check the error messages above.") 