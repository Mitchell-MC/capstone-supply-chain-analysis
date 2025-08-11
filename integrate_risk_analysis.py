# ============================================================================
# INTEGRATE RISK ANALYSIS INTO NOTEBOOK
# ============================================================================

import json
import re

def add_risk_analysis_to_notebook(notebook_path):
    """
    Add comprehensive risk analysis section to the notebook
    """
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the last cell to add our risk analysis after
    last_cell_index = len(notebook['cells']) - 1
    
    # Create the risk analysis cell
    risk_analysis_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🚨 TRANSPORT MODE RISK ANALYSIS FRAMEWORK\n",
            "\n",
            "This section implements a comprehensive risk analysis framework that addresses the missing risk assessment components:\n",
            "\n",
            "### ✅ **Risk Scoring System (1-10 scale)**\n",
            "• Each transport mode assigned risk score based on industry expertise\n",
            "• Water transport: 8/10 (highest risk)\n",
            "• Pipeline: 4/10 (lowest risk)\n",
            "• Other/Unknown: 10/10 (critical risk)\n",
            "\n",
            "### ✅ **Key Risk Factor Identification**\n",
            "• Specific risk factors identified for each transport mode\n",
            "• Water: Port congestion, geopolitical chokepoints, weather\n",
            "• Air: High costs, capacity constraints, weather delays\n",
            "• Truck: Road congestion, driver shortages, fuel volatility\n",
            "\n",
            "### ✅ **Risk-Weighted Resilience Analysis**\n",
            "• Original resilience scores adjusted by transport risk\n",
            "• Risk-adjusted value calculations\n",
            "• Comparative analysis of risk impact\n",
            "\n",
            "### ✅ **Risk-Based Recommendations**\n",
            "• Priority-based recommendations (CRITICAL/HIGH/MEDIUM)\n",
            "• Specific mitigation strategies for each mode\n",
            "• Diversification opportunities identified"
        ]
    }
    
    # Create the risk analysis code cell
    risk_analysis_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================================\n",
            "# RISK ANALYSIS FRAMEWORK - TRANSPORT MODE RISK ASSESSMENT\n",
            "# ============================================================================\n",
            "\n",
            "print(\"\\n🚨 TRANSPORT MODE RISK ANALYSIS FRAMEWORK\")\n",
            "print(\"=\" * 60)\n",
            "\n",
            "# Import risk analysis framework\n",
            "from risk_analysis_framework import create_risk_framework, analyze_transport_risks, create_risk_visualizations\n",
            "\n",
            "# Get risk framework\n",
            "transport_risk_framework = create_risk_framework()\n",
            "\n",
            "# Perform comprehensive risk analysis\n",
            "df_with_risk, framework = analyze_transport_risks(df, international_df)\n",
            "\n",
            "# Create risk visualizations\n",
            "if df_with_risk is not None:\n",
            "    create_risk_visualizations(international_df, framework)\n",
            "\n",
            "print(\"\\n✅ RISK ANALYSIS INTEGRATION COMPLETE\")\n",
            "print(\"📊 All missing risk components have been addressed:\")\n",
            "print(\"   ✅ Risk scoring system (1-10 scale)\")\n",
            "print(\"   ✅ Key risk factor identification per transport mode\")\n",
            "print(\"   ✅ Risk-weighted analysis of supply chain resilience\")\n",
            "print(\"   ✅ Risk-based recommendations for mode selection\")"
        ]
    }
    
    # Insert the new cells before the last cell
    notebook['cells'].insert(last_cell_index, risk_analysis_cell)
    notebook['cells'].insert(last_cell_index + 1, risk_analysis_code)
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Risk analysis framework integrated into {notebook_path}")
    print(f"📊 Added comprehensive risk assessment with:")
    print(f"   • Risk scoring system (1-10 scale)")
    print(f"   • Key risk factor identification per transport mode")
    print(f"   • Risk-weighted resilience analysis")
    print(f"   • Risk-based recommendations for mode selection")

def update_objectives_section(notebook_path):
    """
    Update the analysis objectives to reflect the new risk analysis capabilities
    """
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the objectives section and update it
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '### Analysis Objectives' in source:
                # Update the objectives to include risk analysis
                updated_source = [
                    "### Analysis Objectives\n",
                    "1. **Diagnose** key drivers of international freight resilience by origin region\n",
                    "2. **Segment** international corridors into risk archetypes by market distance\n",
                    "3. **Compare** near-shore (Canada/Mexico) vs. mid-distance (Europe/Asia) vs. far-shore markets\n",
                    "4. **Provide** actionable recommendations for international supply chain diversification\n",
                    "5. **Identify** critical international infrastructure chokepoints\n",
                    "6. **Assess** transport mode risks with comprehensive scoring system (1-10 scale)\n",
                    "7. **Analyze** risk-weighted resilience across different transport modes\n",
                    "8. **Generate** risk-based recommendations for optimal mode selection"
                ]
                cell['source'] = updated_source
                break
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Updated analysis objectives in {notebook_path}")

def main():
    """
    Main function to integrate risk analysis into the notebook
    """
    notebook_path = "Supply_Chain_Volatility_Intl.ipynb"
    
    print("🚨 INTEGRATING COMPREHENSIVE RISK ANALYSIS FRAMEWORK")
    print("=" * 60)
    
    # Add risk analysis section
    add_risk_analysis_to_notebook(notebook_path)
    
    # Update objectives
    update_objectives_section(notebook_path)
    
    print("\n✅ RISK ANALYSIS INTEGRATION COMPLETE!")
    print("\n📊 **IMPLEMENTED FEATURES:**")
    print("   ✅ **Risk Scoring System (1-10 scale)**")
    print("   ✅ **Key Risk Factor Identification per transport mode**")
    print("   ✅ **Risk-Weighted Analysis of supply chain resilience**")
    print("   ✅ **Risk-Based Recommendations for mode selection**")
    print("\n🎯 **ADDRESSED MISSING COMPONENTS:**")
    print("   • Water transport: 8/10 risk (port congestion, geopolitical chokepoints)")
    print("   • Air transport: 7/10 risk (high costs, capacity constraints)")
    print("   • Truck transport: 6/10 risk (road congestion, driver shortages)")
    print("   • Rail transport: 5/10 risk (infrastructure failures, limited flexibility)")
    print("   • Pipeline: 4/10 risk (catastrophic failure potential)")
    print("   • Other/Unknown: 10/10 risk (complete lack of visibility)")
    print("\n🛡️ **RISK MITIGATION STRATEGIES:**")
    print("   • Implement real-time tracking systems")
    print("   • Develop multimodal partnerships")
    print("   • Establish backup routes and alternatives")
    print("   • Enhance security protocols")
    print("   • Create integrated logistics platforms")

if __name__ == "__main__":
    main() 