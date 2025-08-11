# ============================================================================
# TRANSPORT MODE RISK ANALYSIS FRAMEWORK
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_risk_framework():
    """
    Create comprehensive risk framework for transport modes
    """
    transport_risk_framework = {
        1: {
            'name': 'Truck',
            'risk_score': 6,
            'key_risk_factors': [
                'Road congestion and traffic delays',
                'Driver shortages and labor issues', 
                'Fuel price volatility',
                'Cargo theft and security',
                'Weather-related disruptions',
                'Last-mile delivery challenges',
                'Regulatory compliance issues'
            ],
            'risk_mitigation': [
                'Implement real-time tracking systems',
                'Develop driver retention programs',
                'Establish backup routes',
                'Enhance security protocols'
            ]
        },
        2: {
            'name': 'Rail',
            'risk_score': 5,
            'key_risk_factors': [
                'Derailments and infrastructure failures',
                'Limited network flexibility',
                'Intermodal hub delays',
                'Track maintenance issues',
                'Weather-related service disruptions',
                'Labor strikes and disputes'
            ],
            'risk_mitigation': [
                'Invest in infrastructure maintenance',
                'Develop multimodal partnerships',
                'Establish backup rail routes',
                'Implement predictive maintenance'
            ]
        },
        3: {
            'name': 'Water',
            'risk_score': 8,
            'key_risk_factors': [
                'Extremely long and variable lead times',
                'Port congestion and delays',
                'Customs clearance issues',
                'Geopolitical chokepoints',
                'Cargo damage/loss at sea',
                'Weather and natural disasters',
                'Piracy and security threats'
            ],
            'risk_mitigation': [
                'Diversify port options',
                'Implement advanced cargo tracking',
                'Develop air freight alternatives',
                'Establish port partnerships'
            ]
        },
        4: {
            'name': 'Air',
            'risk_score': 7,
            'key_risk_factors': [
                'Extremely high operational costs',
                'Capacity constraints and limited availability',
                'Shipment handling damage',
                'Weather delays at airports',
                'Security screening delays',
                'Fuel price sensitivity',
                'Limited cargo size/weight restrictions'
            ],
            'risk_mitigation': [
                'Optimize cargo consolidation',
                'Develop multi-airport strategies',
                'Implement damage prevention protocols',
                'Establish cost-sharing partnerships'
            ]
        },
        5: {
            'name': 'Multiple modes & mail',
            'risk_score': 7,
            'key_risk_factors': [
                'Complexity at transfer points',
                'Lack of single-point accountability',
                'Potential delays at each hand-off',
                'Increased documentation requirements',
                'Coordination challenges',
                'Cost accumulation across modes'
            ],
            'risk_mitigation': [
                'Implement end-to-end tracking',
                'Establish single-point contact',
                'Develop standardized hand-off procedures',
                'Create integrated logistics platforms'
            ]
        },
        6: {
            'name': 'Pipeline',
            'risk_score': 4,
            'key_risk_factors': [
                'Catastrophic failure potential',
                'Security threats and sabotage',
                'Regulatory compliance hurdles',
                'Environmental liability risks',
                'Limited flexibility for route changes'
            ],
            'risk_mitigation': [
                'Implement advanced monitoring systems',
                'Develop security protocols',
                'Establish regulatory compliance programs',
                'Create backup transport options'
            ]
        },
        7: {
            'name': 'Other and unknown',
            'risk_score': 10,
            'key_risk_factors': [
                'Complete lack of visibility',
                'No control or accountability',
                'Unknown risk factors',
                'Data quality issues',
                'Unpredictable disruptions'
            ],
            'risk_mitigation': [
                'Immediate mode identification required',
                'Implement tracking systems',
                'Establish accountability protocols',
                'Develop contingency plans'
            ]
        },
        8: {
            'name': 'No domestic mode',
            'risk_score': 3,
            'key_risk_factors': [
                'Concentrated dependency on single entry point',
                'Vulnerable to localized disruptions',
                'Limited route alternatives',
                'Regional weather event impacts'
            ],
            'risk_mitigation': [
                'Develop multiple entry points',
                'Establish regional distribution centers',
                'Create backup transport networks',
                'Implement regional risk monitoring'
            ]
        }
    }
    return transport_risk_framework

def analyze_transport_risks(df, international_df):
    """
    Perform comprehensive risk analysis on transport modes
    """
    print("\n🚨 TRANSPORT MODE RISK ANALYSIS FRAMEWORK")
    print("=" * 60)
    
    # Get risk framework
    transport_risk_framework = create_risk_framework()
    
    if len(international_df) > 0:
        print(f"\n📊 RISK ANALYSIS FOR INTERNATIONAL FREIGHT:")
        print(f"   • Total international records: {len(international_df):,}")
        
        # Calculate risk-weighted metrics
        international_df['transport_risk_score'] = international_df['dms_mode'].map(
            {mode: data['risk_score'] for mode, data in transport_risk_framework.items()}
        )
        
        # Calculate risk-weighted value and resilience
        international_df['risk_weighted_value'] = international_df['value_2023_scaled'] * (1 - international_df['transport_risk_score'] / 10)
        
        # Risk analysis by transport mode
        mode_risk_analysis = international_df.groupby('dms_mode').agg({
            'value_2023_scaled': ['sum', 'mean', 'count'],
            'transport_risk_score': 'mean',
            'risk_weighted_value': 'sum'
        }).round(2)
        
        print(f"\n🚛 TRANSPORT MODE RISK ANALYSIS:")
        print(f"{'Mode':<15} {'Records':<10} {'Value ($M)':<12} {'Risk Score':<12} {'Risk-Weighted Value ($M)':<25}")
        print("-" * 80)
        
        for mode_code in mode_risk_analysis.index:
            mode_name = transport_risk_framework.get(mode_code, {}).get('name', f'Mode {mode_code}')
            records = mode_risk_analysis.loc[mode_code, ('value_2023_scaled', 'count')]
            value_millions = mode_risk_analysis.loc[mode_code, ('value_2023_scaled', 'sum')] / 1e6
            risk_score = mode_risk_analysis.loc[mode_code, ('transport_risk_score', 'mean')]
            risk_weighted_value = mode_risk_analysis.loc[mode_code, ('risk_weighted_value', 'sum')] / 1e6
            
            print(f"{mode_name:<15} {records:<10,.0f} ${value_millions:<11,.1f} {risk_score:<12.1f} ${risk_weighted_value:<24,.1f}")
        
        # Identify high-risk transport modes
        high_risk_threshold = 7
        high_risk_modes = international_df[international_df['transport_risk_score'] >= high_risk_threshold]
        
        print(f"\n⚠️  HIGH-RISK TRANSPORT MODES (Risk Score >= {high_risk_threshold}):")
        print(f"   • High-risk records: {len(high_risk_modes):,} ({len(high_risk_modes)/len(international_df)*100:.1f}%)")
        print(f"   • High-risk value: ${high_risk_modes['value_2023_scaled'].sum()/1e9:.1f}B")
        
        if len(high_risk_modes) > 0:
            high_risk_mode_breakdown = high_risk_modes['dms_mode'].value_counts()
            print(f"   • High-risk mode breakdown:")
            for mode_code, count in high_risk_mode_breakdown.items():
                mode_name = transport_risk_framework.get(mode_code, {}).get('name', f'Mode {mode_code}')
                risk_score = transport_risk_framework.get(mode_code, {}).get('risk_score', 'Unknown')
                print(f"     - {mode_name}: {count:,} records (Risk Score: {risk_score}/10)")
        
        # Risk-weighted resilience analysis
        print(f"\n🛡️  RISK-WEIGHTED RESILIENCE ANALYSIS:")
        
        # Calculate risk-adjusted resilience scores
        if 'resilience_score' in international_df.columns:
            international_df['risk_adjusted_resilience'] = international_df['resilience_score'] * (1 - international_df['transport_risk_score'] / 10)
            
            # Compare original vs risk-adjusted resilience
            original_resilience_mean = international_df['resilience_score'].mean()
            risk_adjusted_resilience_mean = international_df['risk_adjusted_resilience'].mean()
            
            print(f"   • Original resilience score: {original_resilience_mean:.2f}")
            print(f"   • Risk-adjusted resilience score: {risk_adjusted_resilience_mean:.2f}")
            print(f"   • Risk impact: {((original_resilience_mean - risk_adjusted_resilience_mean) / original_resilience_mean * 100):.1f}% reduction")
            
            # Risk-adjusted analysis by region
            print(f"\n🌍 RISK-ADJUSTED ANALYSIS BY ORIGIN REGION:")
            risk_by_region = international_df.groupby('origin_foreign_region').agg({
                'resilience_score': 'mean',
                'risk_adjusted_resilience': 'mean',
                'transport_risk_score': 'mean',
                'value_2023_scaled': 'sum'
            }).round(2)
            
            print(f"{'Region':<30} {'Original Resilience':<18} {'Risk-Adjusted':<15} {'Avg Risk Score':<15}")
            print("-" * 80)
            
            for region in risk_by_region.index:
                original = risk_by_region.loc[region, 'resilience_score']
                adjusted = risk_by_region.loc[region, 'risk_adjusted_resilience']
                risk_score = risk_by_region.loc[region, 'transport_risk_score']
                print(f"{region:<30} {original:<18.2f} {adjusted:<15.2f} {risk_score:<15.1f}")
        
        # Risk-based recommendations
        print(f"\n🎯 RISK-BASED RECOMMENDATIONS:")
        
        # Identify most risky modes in use
        agg_columns = {
            'transport_risk_score': 'mean',
            'value_2023_scaled': 'sum'
        }
        
        # Only include resilience_score if it exists
        if 'resilience_score' in international_df.columns:
            agg_columns['resilience_score'] = 'mean'
        
        mode_risk_summary = international_df.groupby('dms_mode').agg(agg_columns).sort_values('transport_risk_score', ascending=False)
        
        recommendations = []
        
        for mode_code in mode_risk_summary.head(3).index:
            mode_data = transport_risk_framework.get(mode_code, {})
            mode_name = mode_data.get('name', f'Mode {mode_code}')
            risk_score = mode_risk_summary.loc[mode_code, 'transport_risk_score']
            value = mode_risk_summary.loc[mode_code, 'value_2023_scaled']
            
            if risk_score >= 8:
                priority = "🔴 CRITICAL"
            elif risk_score >= 6:
                priority = "🟡 HIGH"
            else:
                priority = "🟢 MEDIUM"
            
            recommendations.append(f"{priority} PRIORITY: {mode_name} (Risk Score: {risk_score:.1f}/10)")
            recommendations.append(f"   • Value at risk: ${value/1e6:.1f}M")
            recommendations.append(f"   • Key risk factors: {', '.join(mode_data.get('key_risk_factors', [])[:3])}")
            recommendations.append(f"   • Mitigation: {', '.join(mode_data.get('risk_mitigation', [])[:2])}")
            recommendations.append("")
        
        # Add diversification recommendations
        if len(mode_risk_summary) > 1:
            try:
                # Find the mode with the lowest risk score more safely
                min_risk_score = mode_risk_summary['transport_risk_score'].min()
                min_risk_modes = mode_risk_summary[mode_risk_summary['transport_risk_score'] == min_risk_score]
                
                if len(min_risk_modes) > 0:
                    # Get the first mode with minimum risk score
                    lowest_risk_mode = min_risk_modes.index[0]
                    lowest_risk_name = transport_risk_framework.get(lowest_risk_mode, {}).get('name', f'Mode {lowest_risk_mode}')
                    lowest_risk_score = mode_risk_summary.loc[lowest_risk_mode, 'transport_risk_score']
                    
                    recommendations.append(f"🟢 OPPORTUNITY: Consider increasing {lowest_risk_name} usage")
                    recommendations.append(f"   • Current risk score: {lowest_risk_score:.1f}/10")
                    recommendations.append(f"   • Recommended for high-value, time-sensitive shipments")
            except Exception as e:
                print(f"   ⚠️  Could not determine lowest risk mode: {e}")
                # Continue without diversification recommendation
        
        # Print recommendations
        for rec in recommendations:
            print(f"   {rec}")
        
        # Risk monitoring dashboard metrics
        print(f"\n📊 RISK MONITORING DASHBOARD:")
        print(f"   • Average transport risk score: {international_df['transport_risk_score'].mean():.1f}/10")
        print(f"   • Risk score range: {international_df['transport_risk_score'].min():.1f} - {international_df['transport_risk_score'].max():.1f}")
        print(f"   • High-risk shipments: {len(high_risk_modes):,} ({len(high_risk_modes)/len(international_df)*100:.1f}%)")
        print(f"   • Total value at risk: ${high_risk_modes['value_2023_scaled'].sum()/1e9:.1f}B")
        
        # Risk trend analysis (if temporal data available)
        if 'year' in international_df.columns:
            risk_trends = international_df.groupby('year')['transport_risk_score'].mean()
            if len(risk_trends) > 1:
                print(f"\n📈 RISK TREND ANALYSIS:")
                print(f"   • Risk score trend over time:")
                for year, risk_score in risk_trends.items():
                    print(f"     - {year}: {risk_score:.1f}/10")
        
        print(f"\n✅ RISK ANALYSIS COMPLETE")
        print(f"🔄 Risk metrics update automatically with data changes")
        print(f"📊 Monitor high-risk modes and implement mitigation strategies")
        
        return international_df, transport_risk_framework
        
    else:
        print(f"\n⚠️  NO INTERNATIONAL DATA AVAILABLE FOR RISK ANALYSIS")
        print(f"   • Risk analysis requires international freight records")
        print(f"   • Consider expanding analysis to include international trade data")
        return None, transport_risk_framework

def create_risk_visualizations(international_df, transport_risk_framework):
    """
    Create risk analysis visualizations
    """
    if len(international_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Transport Mode Risk Analysis', fontsize=16, fontweight='bold')
    
    # 1. Risk score distribution by mode
    mode_risk_data = international_df.groupby('dms_mode')['transport_risk_score'].mean()
    mode_names = [transport_risk_framework.get(mode, {}).get('name', f'Mode {mode}') for mode in mode_risk_data.index]
    
    colors = ['#FF6B6B' if score >= 7 else '#4ECDC4' if score >= 5 else '#96CEB4' for score in mode_risk_data.values]
    axes[0,0].bar(mode_names, mode_risk_data.values, color=colors)
    axes[0,0].set_title('Average Risk Score by Transport Mode', fontweight='bold')
    axes[0,0].set_ylabel('Risk Score (1-10)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=7, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
    axes[0,0].legend()
    
    # 2. Value at risk by mode
    mode_value_risk = international_df.groupby('dms_mode').agg({
        'value_2023_scaled': 'sum',
        'transport_risk_score': 'mean'
    })
    mode_value_risk['value_at_risk'] = mode_value_risk['value_2023_scaled'] * (mode_value_risk['transport_risk_score'] / 10)
    
    axes[0,1].bar(mode_names, mode_value_risk['value_at_risk'] / 1e6, color='#FF6B6B')
    axes[0,1].set_title('Value at Risk by Transport Mode', fontweight='bold')
    axes[0,1].set_ylabel('Value at Risk ($M)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Risk vs Value scatter plot
    axes[1,0].scatter(mode_value_risk['transport_risk_score'], 
                      mode_value_risk['value_2023_scaled'] / 1e6,
                      s=100, alpha=0.7, color='#45B7D1')
    
    for i, mode in enumerate(mode_value_risk.index):
        mode_name = transport_risk_framework.get(mode, {}).get('name', f'Mode {mode}')
        axes[1,0].annotate(mode_name, 
                          (mode_value_risk.loc[mode, 'transport_risk_score'], 
                           mode_value_risk.loc[mode, 'value_2023_scaled'] / 1e6),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1,0].set_xlabel('Risk Score (1-10)')
    axes[1,0].set_ylabel('Total Value ($M)')
    axes[1,0].set_title('Risk vs Value Analysis', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Risk-adjusted resilience (if available)
    if 'risk_adjusted_resilience' in international_df.columns:
        risk_resilience_data = international_df.groupby('dms_mode').agg({
            'resilience_score': 'mean',
            'risk_adjusted_resilience': 'mean'
        })
        
        x = np.arange(len(risk_resilience_data))
        width = 0.35
        
        axes[1,1].bar(x - width/2, risk_resilience_data['resilience_score'], 
                      width, label='Original Resilience', color='#4ECDC4')
        axes[1,1].bar(x + width/2, risk_resilience_data['risk_adjusted_resilience'], 
                      width, label='Risk-Adjusted Resilience', color='#FF6B6B')
        
        axes[1,1].set_xlabel('Transport Mode')
        axes[1,1].set_ylabel('Resilience Score')
        axes[1,1].set_title('Original vs Risk-Adjusted Resilience', fontweight='bold')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(mode_names, rotation=45)
        axes[1,1].legend()
    else:
        # Alternative: Risk distribution histogram
        axes[1,1].hist(international_df['transport_risk_score'], bins=10, 
                       color='#45B7D1', alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('Risk Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Risk Score Distribution', fontweight='bold')
        axes[1,1].axvline(x=7, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

# Example usage:
# risk_framework = create_risk_framework()
# df_with_risk, framework = analyze_transport_risks(df, international_df)
# create_risk_visualizations(international_df, framework) 