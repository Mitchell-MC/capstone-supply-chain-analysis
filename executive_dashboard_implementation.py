import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ExecutiveDashboard:
    """
    Executive dashboard for supply chain strategic insights
    """
    
    def __init__(self):
        self.dashboard_data = {}
        self.visualizations = {}
        
    def create_supply_chain_health_score(self, df, risk_scores=None):
        """
        Calculate overall supply chain health score
        """
        print("🏥 Calculating supply chain health score...")
        
        # Base metrics
        efficiency_score = (df['tons_2023'] / (df['tmiles_2023'] + 1)).mean()
        value_density = (df['value_2023'] / (df['tons_2023'] + 1)).mean()
        
        # Risk adjustment (if available)
        risk_adjustment = 0
        if risk_scores is not None:
            risk_adjustment = risk_scores.mean()
        
        # Calculate health score (0-100)
        health_score = (
            efficiency_score * 0.4 +
            value_density * 0.3 +
            (1 - risk_adjustment) * 0.3
        ) * 100
        
        health_score = max(0, min(100, health_score))
        
        # Categorize health level
        if health_score >= 80:
            health_level = "Excellent"
            health_color = "#2E8B57"
        elif health_score >= 60:
            health_level = "Good"
            health_color = "#FFD700"
        elif health_score >= 40:
            health_level = "Fair"
            health_color = "#FF8C00"
        else:
            health_level = "Poor"
            health_color = "#DC143C"
        
        health_metrics = {
            'overall_score': health_score,
            'health_level': health_level,
            'health_color': health_color,
            'efficiency_score': efficiency_score,
            'value_density': value_density,
            'risk_adjustment': risk_adjustment
        }
        
        print(f"   • Overall Health Score: {health_score:.1f}/100 ({health_level})")
        
        return health_metrics
    
    def create_risk_exposure_analysis(self, df, risk_scores=None):
        """
        Analyze risk exposure by region and mode
        """
        print("⚠️ Analyzing risk exposure...")
        
        # Group by region
        regional_risk = df.groupby('fr_orig').agg({
            'tons_2023': 'sum',
            'value_2023': 'sum',
            'tmiles_2023': 'sum'
        }).reset_index()
        
        if risk_scores is not None:
            regional_risk['avg_risk'] = df.groupby('fr_orig')['comprehensive_risk_score'].mean().values
        
        # Group by trade type
        mode_risk = df.groupby('trade_type').agg({
            'tons_2023': 'sum',
            'value_2023': 'sum',
            'tmiles_2023': 'sum'
        }).reset_index()
        
        if risk_scores is not None:
            mode_risk['avg_risk'] = df.groupby('trade_type')['comprehensive_risk_score'].mean().values
        
        risk_analysis = {
            'regional_risk': regional_risk,
            'mode_risk': mode_risk,
            'high_risk_regions': regional_risk[regional_risk['avg_risk'] > regional_risk['avg_risk'].quantile(0.8)] if 'avg_risk' in regional_risk.columns else pd.DataFrame(),
            'high_risk_modes': mode_risk[mode_risk['avg_risk'] > mode_risk['avg_risk'].quantile(0.8)] if 'avg_risk' in mode_risk.columns else pd.DataFrame()
        }
        
        return risk_analysis
    
    def create_cost_impact_analysis(self, df):
        """
        Analyze cost impact and efficiency metrics
        """
        print("💰 Analyzing cost impact...")
        
        # Calculate cost metrics
        df['cost_per_ton'] = df['value_2023'] / (df['tons_2023'] + 1)
        df['cost_per_mile'] = df['value_2023'] / (df['tmiles_2023'] + 1)
        df['efficiency_ratio'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        
        # Cost analysis by region
        regional_cost = df.groupby('fr_orig').agg({
            'cost_per_ton': 'mean',
            'cost_per_mile': 'mean',
            'efficiency_ratio': 'mean',
            'value_2023': 'sum'
        }).reset_index()
        
        # Cost analysis by mode
        mode_cost = df.groupby('trade_type').agg({
            'cost_per_ton': 'mean',
            'cost_per_mile': 'mean',
            'efficiency_ratio': 'mean',
            'value_2023': 'sum'
        }).reset_index()
        
        cost_analysis = {
            'regional_cost': regional_cost,
            'mode_cost': mode_cost,
            'total_value': df['value_2023'].sum(),
            'avg_cost_per_ton': df['cost_per_ton'].mean(),
            'avg_efficiency': df['efficiency_ratio'].mean()
        }
        
        return cost_analysis
    
    def create_strategic_recommendations(self, df, risk_analysis, cost_analysis, health_metrics):
        """
        Generate strategic recommendations based on analysis
        """
        print("💡 Generating strategic recommendations...")
        
        recommendations = []
        
        # Health-based recommendations
        if health_metrics['overall_score'] < 60:
            recommendations.append({
                'category': 'Health Improvement',
                'priority': 'High',
                'recommendation': 'Implement efficiency optimization programs to improve supply chain health score',
                'expected_impact': 'Increase health score by 15-20 points',
                'implementation_time': '6-12 months'
            })
        
        # Risk-based recommendations
        if len(risk_analysis['high_risk_regions']) > 0:
            recommendations.append({
                'category': 'Risk Mitigation',
                'priority': 'High',
                'recommendation': f"Diversify supply sources away from {len(risk_analysis['high_risk_regions'])} high-risk regions",
                'expected_impact': 'Reduce regional concentration risk by 25-30%',
                'implementation_time': '12-18 months'
            })
        
        # Cost-based recommendations
        if cost_analysis['avg_cost_per_ton'] > cost_analysis['regional_cost']['cost_per_ton'].median():
            recommendations.append({
                'category': 'Cost Optimization',
                'priority': 'Medium',
                'recommendation': 'Optimize transportation routes and modes to reduce cost per ton',
                'expected_impact': 'Reduce average cost per ton by 10-15%',
                'implementation_time': '3-6 months'
            })
        
        # Efficiency recommendations
        if cost_analysis['avg_efficiency'] < cost_analysis['mode_cost']['efficiency_ratio'].median():
            recommendations.append({
                'category': 'Efficiency Improvement',
                'priority': 'Medium',
                'recommendation': 'Implement capacity optimization and load consolidation strategies',
                'expected_impact': 'Improve efficiency ratio by 20-25%',
                'implementation_time': '6-9 months'
            })
        
        return recommendations
    
    def create_executive_summary_visualization(self, df, health_metrics, risk_analysis, cost_analysis):
        """
        Create executive summary visualization
        """
        print("📊 Creating executive summary visualization...")
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Supply Chain Health Score', 'Risk Exposure by Region', 
                          'Cost Analysis by Mode', 'Efficiency Distribution'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Health score indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=health_metrics['overall_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Supply Chain Health"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': health_metrics['health_color']},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 60], 'color': "gray"},
                        {'range': [60, 80], 'color': "lightgreen"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Risk exposure by region
        if not risk_analysis['regional_risk'].empty and 'avg_risk' in risk_analysis['regional_risk'].columns:
            fig.add_trace(
                go.Bar(
                    x=risk_analysis['regional_risk']['fr_orig'],
                    y=risk_analysis['regional_risk']['avg_risk'],
                    name='Risk Score',
                    marker_color='red'
                ),
                row=1, col=2
            )
        
        # Cost analysis by mode
        fig.add_trace(
            go.Bar(
                x=risk_analysis['mode_risk']['trade_type'],
                y=risk_analysis['mode_risk']['value_2023'],
                name='Value',
                marker_color='blue'
            ),
            row=2, col=1
        )
        
        # Efficiency distribution
        efficiency_data = df['efficiency_ratio'].dropna()
        fig.add_trace(
            go.Histogram(
                x=efficiency_data,
                nbinsx=20,
                name='Efficiency',
                marker_color='green'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Supply Chain Executive Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_risk_heatmap(self, df, risk_scores=None):
        """
        Create risk heatmap visualization
        """
        print("🔥 Creating risk heatmap...")
        
        # Prepare data for heatmap
        if risk_scores is not None and 'comprehensive_risk_score' in df.columns:
            risk_data = df.groupby(['fr_orig', 'trade_type'])['comprehensive_risk_score'].mean().unstack(fill_value=0)
        else:
            # Create proxy risk based on efficiency
            df['proxy_risk'] = 1 - (df['tons_2023'] / (df['tmiles_2023'] + 1)).rank(pct=True)
            risk_data = df.groupby(['fr_orig', 'trade_type'])['proxy_risk'].mean().unstack(fill_value=0)
        
        # Create heatmap
        fig = px.imshow(
            risk_data,
            title="Risk Heatmap by Region and Trade Type",
            color_continuous_scale="Reds",
            aspect="auto"
        )
        
        fig.update_layout(
            xaxis_title="Trade Type",
            yaxis_title="Origin Region",
            height=600
        )
        
        return fig
    
    def create_cost_efficiency_scatter(self, df):
        """
        Create cost vs efficiency scatter plot
        """
        print("📈 Creating cost-efficiency scatter plot...")
        
        # Calculate metrics
        df['cost_per_ton'] = df['value_2023'] / (df['tons_2023'] + 1)
        df['efficiency_ratio'] = df['tons_2023'] / (df['tmiles_2023'] + 1)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='efficiency_ratio',
            y='cost_per_ton',
            color='trade_type',
            size='value_2023',
            hover_data=['fr_orig', 'fr_dest'],
            title="Cost vs Efficiency Analysis"
        )
        
        fig.update_layout(
            xaxis_title="Efficiency Ratio (tons/mile)",
            yaxis_title="Cost per Ton",
            height=600
        )
        
        return fig
    
    def generate_executive_report(self, df, risk_scores=None):
        """
        Generate comprehensive executive report
        """
        print("📋 Generating executive report...")
        
        # Calculate all metrics
        health_metrics = self.create_supply_chain_health_score(df, risk_scores)
        risk_analysis = self.create_risk_exposure_analysis(df, risk_scores)
        cost_analysis = self.create_cost_impact_analysis(df)
        recommendations = self.create_strategic_recommendations(df, risk_analysis, cost_analysis, health_metrics)
        
        # Create visualizations
        summary_viz = self.create_executive_summary_visualization(df, health_metrics, risk_analysis, cost_analysis)
        risk_heatmap = self.create_risk_heatmap(df, risk_scores)
        cost_efficiency_scatter = self.create_cost_efficiency_scatter(df)
        
        # Compile report
        executive_report = {
            'health_metrics': health_metrics,
            'risk_analysis': risk_analysis,
            'cost_analysis': cost_analysis,
            'recommendations': recommendations,
            'visualizations': {
                'summary': summary_viz,
                'risk_heatmap': risk_heatmap,
                'cost_efficiency': cost_efficiency_scatter
            },
            'summary_stats': {
                'total_corridors': len(df),
                'total_value': df['value_2023'].sum(),
                'total_tons': df['tons_2023'].sum(),
                'avg_distance': df['tmiles_2023'].mean(),
                'high_risk_corridors': len(df[df['comprehensive_risk_score'] > 0.7]) if 'comprehensive_risk_score' in df.columns else 0
            }
        }
        
        # Print executive summary
        print("\n📊 EXECUTIVE SUMMARY:")
        print(f"   • Supply Chain Health: {health_metrics['overall_score']:.1f}/100 ({health_metrics['health_level']})")
        print(f"   • Total Value: ${executive_report['summary_stats']['total_value']/1e9:.1f}B")
        print(f"   • Total Corridors: {executive_report['summary_stats']['total_corridors']:,}")
        print(f"   • High-Risk Corridors: {executive_report['summary_stats']['high_risk_corridors']:,}")
        print(f"   • Strategic Recommendations: {len(recommendations)}")
        
        return executive_report

# Usage function for notebook integration
def implement_executive_dashboard_in_notebook(df, risk_scores=None):
    """
    Function to integrate executive dashboard into the main notebook
    """
    print("📊 IMPLEMENTING EXECUTIVE DASHBOARD")
    print("=" * 50)
    
    # Initialize dashboard
    dashboard = ExecutiveDashboard()
    
    # Generate comprehensive executive report
    executive_report = dashboard.generate_executive_report(df, risk_scores)
    
    return dashboard, executive_report 