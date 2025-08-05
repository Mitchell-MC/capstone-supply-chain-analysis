"""
Integration script to add comprehensive ML enhancements to the Supply Chain Volatility Analysis notebook
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def integrate_ml_enhancements_to_notebook():
    """
    Main function to integrate all ML enhancements into the notebook
    """
    
    print("🚀 INTEGRATING MACHINE LEARNING ENHANCEMENTS")
    print("=" * 60)
    
    # Import all enhancement modules
    try:
        from risk_scoring_implementation import implement_risk_scoring_in_notebook
        from network_analysis_implementation import implement_network_analysis_in_notebook
        from predictive_analytics_implementation import implement_predictive_analytics_in_notebook
        from executive_dashboard_implementation import implement_executive_dashboard_in_notebook
        
        print("✅ All ML enhancement modules loaded successfully")
        
    except ImportError as e:
        print(f"❌ Error loading ML modules: {e}")
        return None
    
    # Create integration code for notebook
    integration_code = """
# ============================================================================
# MACHINE LEARNING ENHANCEMENTS FOR SUPPLY CHAIN ANALYSIS
# ============================================================================

print("🤖 INTEGRATING ADVANCED ML CAPABILITIES")
print("=" * 60)

# Import required libraries for ML enhancements
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Load the enhancement modules
try:
    from risk_scoring_implementation import implement_risk_scoring_in_notebook
    from network_analysis_implementation import implement_network_analysis_in_notebook
    from predictive_analytics_implementation import implement_predictive_analytics_in_notebook
    from executive_dashboard_implementation import implement_executive_dashboard_in_notebook
    print("✅ ML enhancement modules loaded successfully")
except ImportError as e:
    print(f"❌ Error loading ML modules: {e}")
    print("Please ensure all enhancement files are in the same directory")

# ============================================================================
# 1. RISK SCORING SYSTEM IMPLEMENTATION
# ============================================================================

print("\\n🔍 IMPLEMENTING COMPREHENSIVE RISK SCORING SYSTEM")
print("=" * 60)

# Apply risk scoring to international data
if 'international_df' in locals() and len(international_df) > 0:
    try:
        df_with_risk, risk_scorer, high_risk_corridors, risk_report = implement_risk_scoring_in_notebook(international_df)
        
        # Add risk scores to original dataframe
        international_df['comprehensive_risk_score'] = df_with_risk['comprehensive_risk_score']
        
        print("\\n📊 RISK ANALYSIS RESULTS:")
        print(f"   • High-Risk Corridors: {len(high_risk_corridors):,}")
        print(f"   • Average Risk Score: {risk_report['average_risk_score']:.3f}")
        print(f"   • Risk Score Range: {risk_report['min_risk_score']:.3f} - {risk_report['max_risk_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Risk scoring failed: {e}")

# ============================================================================
# 2. NETWORK ANALYSIS & CHOKEPOINT IDENTIFICATION
# ============================================================================

print("\\n🌐 IMPLEMENTING NETWORK ANALYSIS & CHOKEPOINT IDENTIFICATION")
print("=" * 70)

# Apply network analysis to international data
if 'international_df' in locals() and len(international_df) > 0:
    try:
        analyzer, report, G, centrality, chokepoints, vulnerability, clusters, resilience = implement_network_analysis_in_notebook(international_df)
        
        print("\\n🌐 NETWORK ANALYSIS RESULTS:")
        print(f"   • Network Nodes: {report['network_stats']['nodes']:,}")
        print(f"   • Network Edges: {report['network_stats']['edges']:,}")
        print(f"   • Critical Chokepoints: {report['chokepoints']['critical_nodes']:,}")
        print(f"   • Network Density: {report['network_stats']['density']:.3f}")
        print(f"   • Network Clustering Coefficient: {report['network_stats']['clustering_coefficient']:.3f}")
        
    except Exception as e:
        print(f"❌ Network analysis failed: {e}")

# ============================================================================
# 3. PREDICTIVE ANALYTICS FOR DISRUPTION FORECASTING
# ============================================================================

print("\\n🤖 IMPLEMENTING PREDICTIVE ANALYTICS FOR DISRUPTION FORECASTING")
print("=" * 70)

# Apply predictive analytics to international data
if 'international_df' in locals() and len(international_df) > 0:
    try:
        predictor, df_with_predictions, models, cluster_analysis, importance_report, summary = implement_predictive_analytics_in_notebook(international_df)
        
        # Add predictions to original dataframe
        for col in df_with_predictions.columns:
            if 'prediction' in col or 'probability' in col:
                international_df[col] = df_with_predictions[col]
        
        print("\\n🤖 PREDICTIVE ANALYTICS RESULTS:")
        print(f"   • Models Trained: {summary['models_trained']}")
        print(f"   • Disruption Prediction Score: {summary['disruption_prediction_score']:.3f}")
        print(f"   • Cost Prediction R²: {summary['cost_prediction_score']:.3f}")
        print(f"   • Capacity Prediction R²: {summary['capacity_prediction_score']:.3f}")
        print(f"   • Delay Prediction Accuracy: {summary['delay_prediction_score']:.3f}")
        print(f"   • Supply Chain Clusters: {summary['clusters_identified']}")
        
    except Exception as e:
        print(f"❌ Predictive analytics failed: {e}")

# ============================================================================
# 4. EXECUTIVE DASHBOARD CREATION
# ============================================================================

print("\\n📊 IMPLEMENTING EXECUTIVE DASHBOARD")
print("=" * 50)

# Create executive dashboard
if 'international_df' in locals() and len(international_df) > 0:
    try:
        # Get risk scores if available
        risk_scores = None
        if 'comprehensive_risk_score' in international_df.columns:
            risk_scores = international_df['comprehensive_risk_score']
        
        dashboard, executive_report = implement_executive_dashboard_in_notebook(international_df, risk_scores)
        
        print("\\n📊 EXECUTIVE DASHBOARD RESULTS:")
        print(f"   • Supply Chain Health: {executive_report['health_metrics']['overall_score']:.1f}/100")
        print(f"   • Total Value: ${executive_report['summary_stats']['total_value']/1e9:.1f}B")
        print(f"   • Strategic Recommendations: {len(executive_report['recommendations'])}")
        
        # Display strategic recommendations
        print("\\n💡 STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(executive_report['recommendations'], 1):
            print(f"   {i}. {rec['category']} ({rec['priority']} Priority)")
            print(f"      Recommendation: {rec['recommendation']}")
            print(f"      Expected Impact: {rec['expected_impact']}")
            print(f"      Implementation Time: {rec['implementation_time']}")
            print()
        
    except Exception as e:
        print(f"❌ Executive dashboard failed: {e}")

# ============================================================================
# 5. ADVANCED VISUALIZATIONS
# ============================================================================

print("\\n📈 CREATING ADVANCED VISUALIZATIONS")
print("=" * 50)

# Create advanced visualizations if data is available
if 'international_df' in locals() and len(international_df) > 0:
    try:
        # 1. Risk Distribution Visualization
        if 'comprehensive_risk_score' in international_df.columns:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(international_df['comprehensive_risk_score'], bins=30, alpha=0.7, color='red', edgecolor='black')
            plt.title('Risk Score Distribution')
            plt.xlabel('Risk Score')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 2, 2)
            risk_by_region = international_df.groupby('fr_orig')['comprehensive_risk_score'].mean().sort_values(ascending=False)
            plt.bar(range(len(risk_by_region)), risk_by_region.values, color='orange')
            plt.title('Average Risk by Region')
            plt.xlabel('Region')
            plt.ylabel('Average Risk Score')
            
            plt.subplot(2, 2, 3)
            efficiency_vs_risk = plt.scatter(international_df['efficiency_ratio'], 
                                          international_df['comprehensive_risk_score'], 
                                          alpha=0.6, c=international_df['value_2023'], cmap='viridis')
            plt.colorbar(efficiency_vs_risk, label='Value ($)')
            plt.title('Efficiency vs Risk Score')
            plt.xlabel('Efficiency Ratio')
            plt.ylabel('Risk Score')
            
            plt.subplot(2, 2, 4)
            value_by_risk = international_df.groupby(pd.cut(international_df['comprehensive_risk_score'], 5))['value_2023'].sum()
            plt.pie(value_by_risk.values, labels=value_by_risk.index, autopct='%1.1f%%')
            plt.title('Value Distribution by Risk Level')
            
            plt.tight_layout()
            plt.show()
        
        # 2. Predictive Analytics Visualization
        prediction_cols = [col for col in international_df.columns if 'prediction' in col or 'probability' in col]
        if prediction_cols:
            plt.figure(figsize=(15, 10))
            
            for i, col in enumerate(prediction_cols[:4], 1):
                plt.subplot(2, 2, i)
                plt.hist(international_df[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'{col.replace("_", " ").title()}')
                plt.xlabel('Prediction Value')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
        
        print("✅ Advanced visualizations created successfully")
        
    except Exception as e:
        print(f"❌ Advanced visualizations failed: {e}")

# ============================================================================
# 6. MACHINE LEARNING INSIGHTS SUMMARY
# ============================================================================

print("\\n📋 MACHINE LEARNING INSIGHTS SUMMARY")
print("=" * 50)

# Compile ML insights
ml_insights = {
    'risk_analysis': {
        'high_risk_corridors': len(high_risk_corridors) if 'high_risk_corridors' in locals() else 0,
        'avg_risk_score': risk_report['average_risk_score'] if 'risk_report' in locals() else 0
    },
    'network_analysis': {
        'critical_chokepoints': report['chokepoints']['critical_nodes'] if 'report' in locals() else 0,
        'network_density': report['network_stats']['density'] if 'report' in locals() else 0
    },
    'predictive_analytics': {
        'models_trained': summary['models_trained'] if 'summary' in locals() else 0,
        'disruption_prediction_score': summary['disruption_prediction_score'] if 'summary' in locals() else 0
    },
    'executive_dashboard': {
        'health_score': executive_report['health_metrics']['overall_score'] if 'executive_report' in locals() else 0,
        'strategic_recommendations': len(executive_report['recommendations']) if 'executive_report' in locals() else 0
    }
}

print("\\n🎯 KEY ML INSIGHTS:")
print(f"   • Risk Analysis: {ml_insights['risk_analysis']['high_risk_corridors']} high-risk corridors identified")
print(f"   • Network Analysis: {ml_insights['network_analysis']['critical_chokepoints']} critical chokepoints found")
print(f"   • Predictive Analytics: {ml_insights['predictive_analytics']['models_trained']} ML models trained")
print(f"   • Executive Dashboard: {ml_insights['executive_dashboard']['strategic_recommendations']} strategic recommendations generated")

print("\\n✅ MACHINE LEARNING ENHANCEMENTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
"""
    
    return integration_code

def create_notebook_integration_instructions():
    """
    Create instructions for integrating the ML enhancements into the notebook
    """
    
    instructions = """
# ============================================================================
# INTEGRATION INSTRUCTIONS FOR NOTEBOOK ENHANCEMENT
# ============================================================================

## 🚀 How to Integrate ML Enhancements into Your Notebook

### Step 1: Add Required Dependencies
Add these imports at the top of your notebook:

```python
# ML Enhancement Dependencies
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
```

### Step 2: Copy Enhancement Files
Ensure these files are in your notebook directory:
- risk_scoring_implementation.py
- network_analysis_implementation.py
- predictive_analytics_implementation.py
- executive_dashboard_implementation.py

### Step 3: Add Integration Code
Copy the integration code provided by the `integrate_ml_enhancements()` function into a new cell in your notebook.

### Step 4: Run the Integration
Execute the integration code after your data loading and preprocessing steps.

## 🎯 Expected Outcomes

### 1. Risk Scoring System
- Comprehensive risk scores for all supply chain corridors
- Identification of high-risk corridors requiring attention
- Risk prediction models for future planning

### 2. Network Analysis
- Critical chokepoint identification
- Network vulnerability assessment
- Supply chain resilience metrics

### 3. Predictive Analytics
- Disruption prediction models
- Cost forecasting capabilities
- Capacity constraint predictions
- Delivery delay forecasting

### 4. Executive Dashboard
- Supply chain health scoring
- Strategic recommendations
- Interactive visualizations
- Executive summary reports

## 📊 Key Metrics You'll Gain

1. **Risk Metrics:**
   - Comprehensive risk scores (0-1 scale)
   - High-risk corridor identification
   - Risk prediction accuracy

2. **Network Metrics:**
   - Network density and efficiency
   - Critical chokepoint count
   - Vulnerability scores

3. **Predictive Metrics:**
   - Disruption prediction accuracy
   - Cost prediction R² scores
   - Capacity prediction performance

4. **Executive Metrics:**
   - Supply chain health score (0-100)
   - Strategic recommendation count
   - Cost optimization opportunities

## 🔧 Customization Options

### Risk Scoring Weights
Modify the weights in `calculate_comprehensive_risk_score()`:
```python
weights = {
    'geographic': 0.3,    # Geographic concentration risk
    'mode': 0.25,         # Transportation mode risk
    'volatility': 0.25,   # Economic volatility risk
    'infrastructure': 0.2  # Infrastructure vulnerability risk
}
```

### Prediction Model Selection
Choose different ML models in the predictive analytics:
- Random Forest (default)
- Gradient Boosting
- Support Vector Machines
- Neural Networks
- Linear Regression

### Executive Dashboard Thresholds
Adjust health score thresholds:
```python
if health_score >= 80: health_level = "Excellent"
elif health_score >= 60: health_level = "Good"
elif health_score >= 40: health_level = "Fair"
else: health_level = "Poor"
```

## 🎯 Business Impact

### Immediate Benefits:
1. **Risk Mitigation:** Identify and address high-risk corridors
2. **Cost Optimization:** Predict and reduce transportation costs
3. **Efficiency Improvement:** Optimize capacity utilization
4. **Strategic Planning:** Data-driven decision making

### Long-term Benefits:
1. **Resilience Building:** Strengthen supply chain against disruptions
2. **Competitive Advantage:** Predictive capabilities for market advantage
3. **Operational Excellence:** Continuous improvement through ML insights
4. **Executive Visibility:** Clear metrics for strategic oversight

## 📈 Success Metrics

Track these KPIs after implementation:
- Risk score reduction (target: 20-30% improvement)
- Cost per ton reduction (target: 10-15% improvement)
- Supply chain health score (target: 70+)
- Prediction accuracy (target: 80%+)
- Strategic recommendation implementation rate (target: 60%+)

"""
    
    return instructions

if __name__ == "__main__":
    # Generate integration code
    integration_code = integrate_ml_enhancements_to_notebook()
    
    # Generate instructions
    instructions = create_notebook_integration_instructions()
    
    print("✅ Integration script completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Copy the integration code into your notebook")
    print("2. Ensure all enhancement files are in the same directory")
    print("3. Run the integration code after your data preprocessing")
    print("4. Review the ML insights and strategic recommendations") 