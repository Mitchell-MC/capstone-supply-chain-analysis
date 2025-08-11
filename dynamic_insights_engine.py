# ============================================================================
# DYNAMIC INSIGHTS AND RECOMMENDATIONS ENGINE
# ============================================================================

def generate_dynamic_insights(df):
    """
    Generate dynamic insights and recommendations based on current data patterns.
    This function provides real-time analysis that adapts to changing data.
    """
    
    print("DYNAMIC SUPPLY CHAIN INSIGHTS ENGINE")
    print("=" * 60)
    
    # Calculate key metrics for dynamic insights
    total_records = len(df)
    international_records = len(df[df['fr_orig'] >= 800])
    domestic_records = total_records - international_records
    total_value = df['value_2023_scaled'].sum()
    international_value = df[df['fr_orig'] >= 800]['value_2023_scaled'].sum()
    domestic_value = total_value - international_value
    
    # Calculate volatility metrics
    international_df = df[df['fr_orig'] >= 800].copy()
    
    print(f"\nREAL-TIME SUPPLY CHAIN METRICS:")
    print(f"   • Total Supply Chain Value: ${total_value/1e9:.1f}B")
    print(f"   • International Trade Value: ${international_value/1e9:.1f}B ({international_value/total_value*100:.1f}%)")
    print(f"   • Domestic Trade Value: ${domestic_value/1e9:.1f}B ({domestic_value/total_value*100:.1f}%)")
    
    if len(international_df) > 0:
        intl_value_std = international_df['value_2023_scaled'].std()
        intl_value_mean = international_df['value_2023_scaled'].mean()
        intl_volatility = intl_value_std / intl_value_mean if intl_value_mean > 0 else 0
        
        # Mode analysis
        mode_counts = international_df['dms_mode'].value_counts()
        dominant_mode = mode_counts.index[0] if len(mode_counts) > 0 else None
        mode_mapping = {1: 'Truck', 2: 'Rail', 3: 'Water', 4: 'Air', 5: 'Multiple', 6: 'Pipeline', 7: 'Other', 8: 'Unknown'}
        dominant_mode_name = mode_mapping.get(dominant_mode, f'Mode {dominant_mode}') if dominant_mode else 'Unknown'
        
        # Risk assessment
        high_value_threshold = international_df['value_2023_scaled'].quantile(0.9)
        high_value_records = len(international_df[international_df['value_2023_scaled'] > high_value_threshold])
        
        # Efficiency analysis
        international_df['value_per_ton'] = international_df['value_2023_scaled'] / international_df['tons_2023_scaled']
        avg_value_per_ton = international_df['value_per_ton'].mean()
        
        # Distance analysis (using fr_orig as proxy for distance)
        near_shore = international_df[international_df['fr_orig'].isin([801, 802])]  # Canada, Mexico
        mid_distance = international_df[international_df['fr_orig'].isin([803, 804, 805])]  # Europe, Asia
        far_shore = international_df[~international_df['fr_orig'].isin([801, 802, 803, 804, 805])]
        
        near_shore_value = near_shore['value_2023_scaled'].sum() if len(near_shore) > 0 else 0
        mid_distance_value = mid_distance['value_2023_scaled'].sum() if len(mid_distance) > 0 else 0
        far_shore_value = far_shore['value_2023_scaled'].sum() if len(far_shore) > 0 else 0
        
        print(f"\nINTERNATIONAL TRADE INSIGHTS:")
        print(f"   • International Records: {international_records:,} ({international_records/total_records*100:.1f}% of total)")
        print(f"   • Average International Shipment Value: ${international_df['value_2023_scaled'].mean():,.0f}")
        print(f"   • International Value Volatility: {intl_volatility:.2f} (Higher = More Volatile)")
        
        # Dynamic recommendations based on volatility
        if intl_volatility > 2.0:
            print(f"   WARNING: HIGH VOLATILITY DETECTED: International trade shows significant volatility")
            print(f"      RECOMMENDATION: Implement risk mitigation strategies and diversify suppliers")
            print(f"      ACTION: Consider near-shoring and multi-sourcing strategies")
        elif intl_volatility > 1.0:
            print(f"   WARNING: MODERATE VOLATILITY: International trade shows moderate volatility")
            print(f"      RECOMMENDATION: Monitor trends and consider backup suppliers")
            print(f"      ACTION: Develop contingency plans for key supply routes")
        else:
            print(f"   SUCCESS: STABLE TRADE: International trade shows low volatility")
            print(f"      RECOMMENDATION: Current strategy appears stable")
            print(f"      ACTION: Continue monitoring and optimize existing routes")
        
        print(f"\nTRANSPORT MODE ANALYSIS:")
        print(f"   • Dominant Mode: {dominant_mode_name} ({mode_counts.iloc[0]:,} shipments)")
        print(f"   • Mode Diversity: {len(mode_counts)} different transport modes")
        
        # Dynamic transport recommendations
        if dominant_mode == 1:  # Truck
            print(f"      RECOMMENDATION: Truck dominance suggests regional trade focus")
            print(f"      OPPORTUNITY: Consider rail for longer distances to reduce costs")
            print(f"      ACTION: Evaluate rail infrastructure for cost optimization")
        elif dominant_mode == 2:  # Rail
            print(f"      RECOMMENDATION: Rail dominance indicates efficient long-distance transport")
            print(f"      OPPORTUNITY: Consider multimodal options for last-mile delivery")
            print(f"      ACTION: Develop truck-rail partnerships for seamless delivery")
        elif dominant_mode == 3:  # Water
            print(f"      RECOMMENDATION: Water transport suggests cost-effective bulk shipping")
            print(f"      OPPORTUNITY: Consider faster modes for time-sensitive goods")
            print(f"      ACTION: Balance cost vs. speed for different product categories")
        elif dominant_mode == 4:  # Air
            print(f"      RECOMMENDATION: Air transport indicates high-value, time-sensitive goods")
            print(f"      OPPORTUNITY: Consider cost optimization for non-urgent shipments")
            print(f"      ACTION: Implement tiered shipping strategies based on urgency")
        
        print(f"\nREGIONAL TRADE PATTERNS:")
        print(f"   • Near-Shore Trade (Canada/Mexico): ${near_shore_value/1e9:.1f}B")
        print(f"   • Mid-Distance Trade (Europe/Asia): ${mid_distance_value/1e9:.1f}B")
        print(f"   • Far-Shore Trade (Other Regions): ${far_shore_value/1e9:.1f}B")
        
        # Dynamic regional recommendations
        if near_shore_value > mid_distance_value and near_shore_value > far_shore_value:
            print(f"      RECOMMENDATION: Near-shore trade dominates - good for resilience")
            print(f"      OPPORTUNITY: Consider expanding mid-distance trade for diversification")
            print(f"      ACTION: Develop strategic partnerships in Europe and Asia")
        elif mid_distance_value > near_shore_value and mid_distance_value > far_shore_value:
            print(f"      RECOMMENDATION: Mid-distance trade dominates - balanced approach")
            print(f"      OPPORTUNITY: Consider increasing near-shore trade for risk mitigation")
            print(f"      ACTION: Strengthen trade relationships with Canada and Mexico")
        else:
            print(f"      RECOMMENDATION: Far-shore trade significant - consider diversification")
            print(f"      OPPORTUNITY: Increase near-shore and mid-distance trade for resilience")
            print(f"      ACTION: Develop regional supply chain hubs for risk mitigation")
        
        print(f"\nVALUE DENSITY ANALYSIS:")
        print(f"   • Average Value per Ton: ${avg_value_per_ton:.2f}")
        print(f"   • High-Value Shipments (>90th percentile): {high_value_records:,} records")
        
        # Dynamic value recommendations
        if avg_value_per_ton > 1000:
            print(f"      RECOMMENDATION: High-value density suggests premium goods")
            print(f"      OPPORTUNITY: Consider air transport for time-sensitive high-value items")
            print(f"      ACTION: Implement premium shipping lanes for high-value products")
        elif avg_value_per_ton > 100:
            print(f"      RECOMMENDATION: Moderate value density - balanced approach")
            print(f"      OPPORTUNITY: Optimize transport mode based on distance and urgency")
            print(f"      ACTION: Develop flexible routing based on market conditions")
        else:
            print(f"      RECOMMENDATION: Low value density suggests bulk commodities")
            print(f"      OPPORTUNITY: Focus on cost-effective transport modes")
            print(f"      ACTION: Optimize bulk shipping routes for cost efficiency")
        
        print(f"\nSTRATEGIC RECOMMENDATIONS:")
        
        # Generate dynamic recommendations based on data patterns
        recommendations = []
        
        if intl_volatility > 1.5:
            recommendations.append("HIGH PRIORITY: Implement supply chain risk mitigation strategies")
            recommendations.append("HIGH PRIORITY: Diversify supplier base across multiple regions")
            recommendations.append("HIGH PRIORITY: Develop real-time monitoring systems")
        
        if near_shore_value < total_value * 0.3:
            recommendations.append("MEDIUM PRIORITY: Increase near-shore trade for resilience")
            recommendations.append("MEDIUM PRIORITY: Develop regional supply chain hubs")
        
        if dominant_mode == 1 and international_records > 1000:
            recommendations.append("MEDIUM PRIORITY: Consider multimodal transport for international routes")
            recommendations.append("MEDIUM PRIORITY: Evaluate rail and water alternatives")
        
        if high_value_records > international_records * 0.1:
            recommendations.append("LOW PRIORITY: Monitor high-value shipments for security")
            recommendations.append("LOW PRIORITY: Implement tracking systems for premium goods")
        
        if len(recommendations) == 0:
            recommendations.append("SUCCESS: SUPPLY CHAIN APPEARS WELL-BALANCED")
            recommendations.append("SUCCESS: CONTINUE CURRENT STRATEGY WITH REGULAR MONITORING")
            recommendations.append("SUCCESS: MAINTAIN EXISTING PARTNERSHIPS AND ROUTES")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"   • Data Quality Score: {min(100, (1 - len(df[df['value_2023_scaled'] == 0])/len(df)) * 100:.1f}%")
        print(f"   • International Coverage: {len(international_df['fr_orig'].unique())} origin regions")
        print(f"   • Transport Mode Coverage: {len(mode_counts)} modes")
        print(f"   • Value Distribution: {international_df['value_2023_scaled'].skew():.2f} skewness")
        
        # Additional dynamic insights
        print(f"\nADDITIONAL INSIGHTS:")
        
        # Market concentration analysis
        top_5_regions = international_df.groupby('fr_orig')['value_2023_scaled'].sum().sort_values(ascending=False).head(5)
        top_5_percentage = top_5_regions.sum() / international_value * 100
        
        if top_5_percentage > 80:
            print(f"   WARNING: HIGH CONCENTRATION: Top 5 regions account for {top_5_percentage:.1f}% of international trade")
            print(f"      RECOMMENDATION: Diversify trade partners to reduce concentration risk")
        elif top_5_percentage > 60:
            print(f"   WARNING: MODERATE CONCENTRATION: Top 5 regions account for {top_5_percentage:.1f}% of international trade")
            print(f"      RECOMMENDATION: Consider expanding trade with emerging markets")
        else:
            print(f"   SUCCESS: GOOD DIVERSIFICATION: Top 5 regions account for {top_5_percentage:.1f}% of international trade")
            print(f"      RECOMMENDATION: Maintain current diversification strategy")
        
        # Seasonal or temporal patterns (if time data available)
        if 'year' in df.columns:
            year_counts = international_df['year'].value_counts()
            if len(year_counts) > 1:
                print(f"   TEMPORAL ANALYSIS: Data spans {len(year_counts)} years")
                print(f"      RECOMMENDATION: Monitor year-over-year trends for seasonal patterns")
        
        print(f"\nDYNAMIC ANALYSIS COMPLETE")
        print(f"Recommendations update automatically based on data changes")
        print(f"Run this function again after data updates for fresh insights")
        print(f"Use these insights to inform strategic supply chain decisions")
        
    else:
        print(f"\nWARNING: NO INTERNATIONAL DATA FOUND")
        print(f"   • All {total_records:,} records are domestic")
        print(f"   • Consider expanding analysis to include international trade data")
        print(f"   RECOMMENDATION: Develop international trade capabilities")
        print(f"   ACTION: Identify potential international trade partners")

# Example usage:
# generate_dynamic_insights(df) 