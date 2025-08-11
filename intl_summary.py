import argparse
import pickle
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
from risk_analysis_framework import create_risk_framework


def generate_notebook_insights(df: pd.DataFrame) -> None:
    """
    Print dynamic insights and recommendations based on the provided dataframe.
    This consolidates notebook logic into a reusable script function.
    """
    print("🚀 DYNAMIC SUPPLY CHAIN INSIGHTS ENGINE")
    print("=" * 60)

    total_records = len(df)
    international_df = df[df.get('fr_orig', pd.Series(index=df.index)).fillna(0) >= 800].copy()

    total_value = df.get('value_2023_scaled', pd.Series(0, index=df.index)).sum()
    international_value = international_df.get('value_2023_scaled', pd.Series(0, index=international_df.index)).sum()
    domestic_value = total_value - international_value
    international_records = len(international_df)

    print(f"\n📊 REAL-TIME SUPPLY CHAIN METRICS:")
    print(f"   • Total Supply Chain Value: ${total_value/1e9:.1f}B")
    print(f"   • International Trade Value: ${international_value/1e9:.1f}B ({(international_value/total_value*100 if total_value else 0):.1f}%)")
    print(f"   • Domestic Trade Value: ${domestic_value/1e9:.1f}B ({(domestic_value/total_value*100 if total_value else 0):.1f}%)")

    if len(international_df) == 0:
        print(f"\n⚠️  NO INTERNATIONAL DATA FOUND")
        print(f"   • All {total_records:,} records are domestic")
        print(f"   • Consider expanding analysis to include international trade data")
        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"🔄 Recommendations update automatically based on data changes")
        return

    intl_value_std = international_df['value_2023_scaled'].std()
    intl_value_mean = international_df['value_2023_scaled'].mean()
    intl_volatility = (intl_value_std / intl_value_mean) if intl_value_mean > 0 else 0

    mode_counts = international_df['dms_mode'].value_counts()
    dominant_mode = mode_counts.index[0] if len(mode_counts) > 0 else None
    mode_mapping = {1: 'Truck', 2: 'Rail', 3: 'Water', 4: 'Air', 5: 'Multiple', 6: 'Pipeline', 7: 'Other', 8: 'Unknown'}
    dominant_mode_name = mode_mapping.get(dominant_mode, f'Mode {dominant_mode}') if dominant_mode else 'Unknown'

    near_shore = international_df[international_df['fr_orig'].isin([801, 802])]
    mid_distance = international_df[international_df['fr_orig'].isin([803, 804, 805])]
    far_shore = international_df[~international_df['fr_orig'].isin([801, 802, 803, 804, 805])]

    near_shore_value = near_shore['value_2023_scaled'].sum() if len(near_shore) > 0 else 0.0
    mid_distance_value = mid_distance['value_2023_scaled'].sum() if len(mid_distance) > 0 else 0.0
    far_shore_value = far_shore['value_2023_scaled'].sum() if len(far_shore) > 0 else 0.0

    high_value_threshold = international_df['value_2023_scaled'].quantile(0.9)
    high_value_records = len(international_df[international_df['value_2023_scaled'] > high_value_threshold])

    international_df['value_per_ton'] = international_df['value_2023_scaled'] / international_df['tons_2023_scaled']
    avg_value_per_ton = international_df['value_per_ton'].mean()

    print(f"\n🌍 INTERNATIONAL TRADE INSIGHTS:")
    print(f"   • International Records: {international_records:,} ({(international_records/total_records*100 if total_records else 0):.1f}% of total)")
    print(f"   • Average International Shipment Value: ${international_df['value_2023_scaled'].mean():,.0f}")
    print(f"   • International Value Volatility: {intl_volatility:.2f} (Higher = More Volatile)")

    if intl_volatility > 2.0:
        print(f"   ⚠️  HIGH VOLATILITY DETECTED: International trade shows significant volatility")
        print(f"      💡 RECOMMENDATION: Implement risk mitigation strategies and diversify suppliers")
    elif intl_volatility > 1.0:
        print(f"   ⚠️  MODERATE VOLATILITY: International trade shows moderate volatility")
        print(f"      💡 RECOMMENDATION: Monitor trends and consider backup suppliers")
    else:
        print(f"   ✅ STABLE TRADE: International trade shows low volatility")
        print(f"      💡 RECOMMENDATION: Current strategy appears stable")

    print(f"\n🚛 TRANSPORT MODE ANALYSIS:")
    print(f"   • Dominant Mode: {dominant_mode_name} ({(mode_counts.iloc[0] if len(mode_counts)>0 else 0):,} shipments)")
    print(f"   • Mode Diversity: {len(mode_counts)} different transport modes")

    if dominant_mode == 1:
        print(f"      💡 RECOMMENDATION: Truck dominance suggests regional trade focus")
        print(f"      💡 OPPORTUNITY: Consider rail for longer distances to reduce costs")
    elif dominant_mode == 2:
        print(f"      💡 RECOMMENDATION: Rail dominance indicates efficient long-distance transport")
        print(f"      💡 OPPORTUNITY: Consider multimodal options for last-mile delivery")
    elif dominant_mode == 3:
        print(f"      💡 RECOMMENDATION: Water transport suggests cost-effective bulk shipping")
        print(f"      💡 OPPORTUNITY: Consider faster modes for time-sensitive goods")
    elif dominant_mode == 4:
        print(f"      💡 RECOMMENDATION: Air transport indicates high-value, time-sensitive goods")
        print(f"      💡 OPPORTUNITY: Consider cost optimization for non-urgent shipments")

    print(f"\n🌍 REGIONAL TRADE PATTERNS:")
    print(f"   • Near-Shore Trade (Canada/Mexico): ${near_shore_value/1e9:.1f}B")
    print(f"   • Mid-Distance Trade (Europe/Asia): ${mid_distance_value/1e9:.1f}B")
    print(f"   • Far-Shore Trade (Other Regions): ${far_shore_value/1e9:.1f}B")

    if near_shore_value > mid_distance_value and near_shore_value > far_shore_value:
        print(f"      💡 RECOMMENDATION: Near-shore trade dominates - good for resilience")
        print(f"      💡 OPPORTUNITY: Consider expanding mid-distance trade for diversification")
    elif mid_distance_value > near_shore_value and mid_distance_value > far_shore_value:
        print(f"      💡 RECOMMENDATION: Mid-distance trade dominates - balanced approach")
        print(f"      💡 OPPORTUNITY: Consider increasing near-shore trade for risk mitigation")
    else:
        print(f"      💡 RECOMMENDATION: Far-shore trade significant - consider diversification")
        print(f"      💡 OPPORTUNITY: Increase near-shore and mid-distance trade for resilience")

    print(f"\n💰 VALUE DENSITY ANALYSIS:")
    print(f"   • Average Value per Ton: ${avg_value_per_ton:.2f}")
    print(f"   • High-Value Shipments (>90th percentile): {high_value_records:,} records")

    if avg_value_per_ton > 1000:
        print(f"      💡 RECOMMENDATION: High-value density suggests premium goods")
        print(f"      💡 OPPORTUNITY: Consider air transport for time-sensitive high-value items")
    elif avg_value_per_ton > 100:
        print(f"      💡 RECOMMENDATION: Moderate value density - balanced approach")
        print(f"      💡 OPPORTUNITY: Optimize transport mode based on distance and urgency")
    else:
        print(f"      💡 RECOMMENDATION: Low value density suggests bulk commodities")
        print(f"      💡 OPPORTUNITY: Focus on cost-effective transport modes")

    print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
    recs: list[str] = []
    if intl_volatility > 1.5:
        recs.append("🔴 HIGH PRIORITY: Implement supply chain risk mitigation strategies")
        recs.append("🔴 HIGH PRIORITY: Diversify supplier base across multiple regions")
    if near_shore_value < total_value * 0.3:
        recs.append("🟡 MEDIUM PRIORITY: Increase near-shore trade for resilience")
    if dominant_mode == 1 and international_records > 1000:
        recs.append("🟡 MEDIUM PRIORITY: Consider multimodal transport for international routes")
    if high_value_records > international_records * 0.1:
        recs.append("🟢 LOW PRIORITY: Monitor high-value shipments for security")
    if not recs:
        recs.append("✅ SUPPLY CHAIN APPEARS WELL-BALANCED")
        recs.append("✅ CONTINUE CURRENT STRATEGY WITH REGULAR MONITORING")
    for i, rec in enumerate(recs, 1):
        print(f"   {i}. {rec}")

    print(f"\n📈 PERFORMANCE METRICS:")
    data_quality_score = 0.0
    try:
        data_quality_score = min(100.0, (1 - len(df[df['value_2023_scaled'] == 0]) / max(1, len(df))) * 100)
    except Exception:
        pass
    print(f"   • Data Quality Score: {data_quality_score:.1f}%")
    try:
        print(f"   • International Coverage: {len(international_df['fr_orig'].unique())} origin regions")
    except Exception:
        pass
    print(f"   • Transport Mode Coverage: {len(mode_counts)} modes")
    try:
        print(f"   • Value Distribution: {international_df['value_2023_scaled'].skew():.2f} skewness")
    except Exception:
        pass

    print(f"\n✅ ANALYSIS COMPLETE")
    print(f"🔄 Recommendations update automatically based on data changes")


def normalize_mode_name(mode_name: str) -> str:
    if 'Multiple modes' in mode_name:
        return 'Multiple modes'
    if 'Other' in mode_name:
        return 'Other'
    return mode_name


def format_money(value: float) -> str:
    try:
        return f"${value:,.0f}"
    except Exception:
        return "$0"


def format_ratio(value: float) -> str:
    try:
        return f"{value:.2f}"
    except Exception:
        return "N/A"


def build_summary(payload: Dict[str, Any]) -> None:
    international_df: pd.DataFrame = payload.get('international_df')
    ccc_analysis: pd.DataFrame | None = payload.get('ccc_analysis')

    if international_df is None or not isinstance(international_df, pd.DataFrame) or international_df.empty:
        print("No international_df provided or it is empty.")
        return

    # Ensure derived columns
    if (
        'value_density' not in international_df.columns
        and 'value_2023' in international_df.columns
        and 'tons_2023' in international_df.columns
    ):
        international_df = international_df.copy()
        international_df['value_density'] = international_df['value_2023'] / (international_df['tons_2023'] + 1e-6)

    # Market comparison by distance if available
    group_keys = [key for key in ['origin_market_distance'] if key in international_df.columns]
    if group_keys:
        market_comparison = international_df.groupby(group_keys[0]).agg({
            'tons_2023': 'sum',
            'value_2023': 'sum',
            'resilience_score': 'mean',
            'tons_volatility': 'mean',
            'value_density': 'mean'
        }).round(4)
    else:
        market_comparison = pd.DataFrame()

    def safe_idxmax(series: pd.Series):
        try:
            return series.idxmax()
        except Exception:
            return None

    def safe_idxmin(series: pd.Series):
        try:
            return series.idxmin()
        except Exception:
            return None

    best_resilience_market = safe_idxmax(market_comparison['resilience_score']) if not market_comparison.empty else None
    worst_resilience_market = safe_idxmin(market_comparison['resilience_score']) if not market_comparison.empty else None

    best_value_market = safe_idxmax(market_comparison['value_density']) if not market_comparison.empty else None
    worst_value_market = safe_idxmin(market_comparison['value_density']) if not market_comparison.empty else None

    lowest_vol_market = safe_idxmin(market_comparison['tons_volatility']) if not market_comparison.empty else None
    highest_vol_market = safe_idxmax(market_comparison['tons_volatility']) if not market_comparison.empty else None

    # High-risk cohort (bottom quartile by resilience)
    q25 = international_df['resilience_score'].quantile(0.25) if 'resilience_score' in international_df.columns else np.nan
    high_risk = (
        international_df[international_df['resilience_score'] <= q25]
        if pd.notna(q25)
        else international_df.iloc[[]]
    )
    high_risk_share = (len(high_risk) / max(1, len(international_df))) * 100 if len(international_df) else 0.0
    high_risk_tons = high_risk['tons_2023'].sum() if 'tons_2023' in high_risk.columns else 0.0
    high_risk_value = high_risk['value_2023'].sum() if 'value_2023' in high_risk.columns else 0.0

    # Mode mapping and risk-reward
    correct_mode_mapping = {
        1: 'Truck', 2: 'Rail', 3: 'Water', 4: 'Air',
        5: 'Multiple modes & mail', 6: 'Pipeline',
        7: 'Other and unknown', 8: 'No domestic mode'
    }

    mode_risk_scores = {
        'Truck': 6, 'Rail': 5, 'Water': 8, 'Air': 7,
        'Pipeline': 4, 'Multiple modes': 6, 'Other': 10, 'No domestic mode': 6
    }

    if 'dms_mode' in international_df.columns:
        mode_resilience = international_df.groupby('dms_mode')['resilience_score'].mean()
        best_mode_code = mode_resilience.idxmax() if len(mode_resilience) else None
        best_mode_name = (
            correct_mode_mapping.get(best_mode_code, f"Mode {best_mode_code}")
            if best_mode_code is not None else "N/A"
        )
        best_mode_resilience = mode_resilience.get(best_mode_code, np.nan)

        mode_risk_reward: Dict[str, float] = {}
        for mode_code, resilience in mode_resilience.items():
            name = normalize_mode_name(correct_mode_mapping.get(mode_code, f"Mode {mode_code}"))
            risk = mode_risk_scores.get(name, 6)
            mode_risk_reward[name] = resilience / risk if risk > 0 else np.nan

        best_rr_mode = max(mode_risk_reward, key=mode_risk_reward.get) if len(mode_risk_reward) else "N/A"
        best_rr_value = mode_risk_reward.get(best_rr_mode, np.nan)
    else:
        best_mode_name = "N/A"
        best_mode_resilience = np.nan
        best_rr_mode = "N/A"
        best_rr_value = np.nan

    # Optional CCC insight
    ccc_lines: list[str] = []
    if (
        isinstance(ccc_analysis, pd.DataFrame)
        and 'weighted_ccc' in ccc_analysis.columns
        and len(ccc_analysis)
    ):
        best_ccc_market = ccc_analysis['weighted_ccc'].idxmin()
        worst_ccc_market = ccc_analysis['weighted_ccc'].idxmax()
        ccc_lines.append(
            f"In cash conversion terms, {str(best_ccc_market).lower()} corridors turn cash fastest, while "
            f"{str(worst_ccc_market).lower()} take the longest to convert shipments into cash."
        )

    # Executive narrative
    num_records = len(international_df)
    resilience_mean = international_df['resilience_score'].mean() if 'resilience_score' in international_df.columns else float('nan')
    resilience_min = international_df['resilience_score'].min() if 'resilience_score' in international_df.columns else float('nan')
    resilience_max = international_df['resilience_score'].max() if 'resilience_score' in international_df.columns else float('nan')

    segments: list[str] = []

    segments.append(
        f"Across {num_records:,} international records, the network shows a mean resilience score of "
        f"{resilience_mean:.1f} (range {resilience_min:.1f}–{resilience_max:.1f}). "
        f"{str(best_resilience_market)} corridors are generally the steadiest, while "
        f"{str(worst_resilience_market)} corridors tend to lag and would benefit from focused improvement."
    )

    if best_value_market is not None and worst_value_market is not None and not market_comparison.empty:
        segments.append(
            f"From a value-density perspective, {str(best_value_market).lower()} markets deliver the richest dollars per ton "
            f"({format_ratio(float(market_comparison.loc[best_value_market, 'value_density']))}/ton), "
            f"outpacing {str(worst_value_market).lower()} lanes, which are relatively thinner on value."
        )

    if lowest_vol_market is not None and highest_vol_market is not None:
        segments.append(
            f"Volatility is lowest in {str(lowest_vol_market).lower()} corridors and highest in "
            f"{str(highest_vol_market).lower()}, suggesting where to expect steadier planning versus where to build buffers."
        )

    segments.append(
        f"Roughly {high_risk_share:.1f}% of international lanes sit in the bottom resilience quartile today, "
        f"representing about {high_risk_tons:,.0f} tons and {format_money(high_risk_value)} in value at risk. "
        f"These lanes warrant closer monitoring and targeted stabilization."
    )

    if best_mode_name != "N/A" and not np.isnan(best_mode_resilience):
        segments.append(
            f"By mode, {best_mode_name.lower()} currently leads on resilience (≈{best_mode_resilience:.1f}). "
            f"After accounting for inherent mode risk, the best risk‑reward balance is with "
            f"{best_rr_mode.lower()} (ratio ≈ {best_rr_value:.2f})."
        )

    segments.extend(ccc_lines)

    # Dynamic high-risk mode diversification recommendation
    recommendations: list[str] = []

    if 'dms_mode' in international_df.columns and 'resilience_score' in international_df.columns:
        high_risk_modes = [m for m, r in {
            'Truck': 6, 'Rail': 5, 'Water': 8, 'Air': 7,
            'Pipeline': 4, 'Multiple modes': 6, 'Other': 10, 'No domestic mode': 6
        }.items() if r >= 7]

        df2 = international_df.copy()
        df2['mode_name'] = df2['dms_mode'].map(correct_mode_mapping).fillna('Other')
        df2['mode_name_norm'] = df2['mode_name'].apply(normalize_mode_name)

        threshold_value = df2['value_2023'].quantile(0.90) if 'value_2023' in df2.columns else float('inf')
        critical_mask = (
            (df2['resilience_score'] <= q25) if pd.notna(q25) else pd.Series(False, index=df2.index)
        ) | (
            (df2['value_2023'] >= threshold_value) if 'value_2023' in df2.columns else pd.Series(False, index=df2.index)
        )
        high_risk_mode_mask = df2['mode_name_norm'].isin(high_risk_modes)

        critical_high_risk = df2[critical_mask & high_risk_mode_mask].copy()
        total_value = max(1.0, float(df2['value_2023'].sum())) if 'value_2023' in df2.columns else 1.0
        share_value = (float(critical_high_risk['value_2023'].sum()) / total_value) * 100 if 'value_2023' in critical_high_risk.columns else 0.0

        origin_col = 'origin_foreign_region' if 'origin_foreign_region' in df2.columns else None
        dest_col = 'dest_state_name' if 'dest_state_name' in df2.columns else None
        if origin_col and dest_col:
            route_key = (
                df2.loc[critical_high_risk.index, origin_col].fillna('Unknown').astype(str)
                + " → " + df2.loc[critical_high_risk.index, dest_col].fillna('Unknown').astype(str)
            )
            num_critical_routes = route_key.nunique()
        else:
            num_critical_routes = int(critical_high_risk.shape[0])

        # Alternatives: top risk-reward modes excluding high-risk ones
        mode_risk_reward_map: Dict[str, float] = {}
        if 'dms_mode' in international_df.columns:
            mode_resilience = international_df.groupby('dms_mode')['resilience_score'].mean()
            for mode_code, resilience in mode_resilience.items():
                name = normalize_mode_name(correct_mode_mapping.get(mode_code, f"Mode {mode_code}"))
                risk = mode_risk_scores.get(name, 6)
                mode_risk_reward_map[name] = resilience / risk if risk > 0 else np.nan

        alternatives_sorted = sorted(
            [(m, v) for m, v in mode_risk_reward_map.items() if m not in high_risk_modes and np.isfinite(v)],
            key=lambda x: x[1],
            reverse=True
        )
        alternative_modes = [m for m, _ in alternatives_sorted[:2]] or [best_rr_mode]

        dynamic_recommendation = (
            f"As we balance cost and resilience, favor {best_rr_mode.lower()} for its superior risk‑reward profile. "
            f"For critical lanes—low‑resilience or high‑value routes—dial back dependence on "
            f"{', '.join([m.lower() for m in high_risk_modes])} where they dominate today: roughly {num_critical_routes:,} key lanes "
            f"exposed (~{share_value:.1f}% of international value). Where feasible, pivot toward "
            f"{' and '.join([m.lower() for m in alternative_modes])} to reduce risk without sacrificing throughput."
        )
        recommendations.append(dynamic_recommendation)

    # Output
    print("\nINTERNATIONAL ANALYSIS — EXECUTIVE SUMMARY\n" + "-" * 60)
    for segment in segments:
        print(segment + "\n")

    print("RECOMMENDATIONS\n" + "-" * 60)
    for rec in recommendations:
        print(rec + "\n")

    # Risk-based mode recommendations using transport risk framework
    try:
        if 'dms_mode' in international_df.columns:
            transport_risk_framework = create_risk_framework()
            mode_to_risk = {mode: data.get('risk_score', np.nan) for mode, data in transport_risk_framework.items()}

            df_modes = international_df.copy()
            df_modes['transport_risk_score'] = df_modes['dms_mode'].map(mode_to_risk)

            # Pick value column (fallback to counts if not present)
            value_col = 'value_2023' if 'value_2023' in df_modes.columns else (
                'value_2023_scaled' if 'value_2023_scaled' in df_modes.columns else None
            )

            agg_dict: Dict[str, str] = {'transport_risk_score': 'mean'}
            if value_col:
                agg_dict[value_col] = 'sum'

            mode_summary = df_modes.groupby('dms_mode').agg(agg_dict)
            mode_summary = mode_summary.sort_values('transport_risk_score', ascending=False)

            print("RISK-BASED RECOMMENDATIONS\n" + "-" * 60)

            for mode_code in list(mode_summary.index)[:3]:
                mode_data = transport_risk_framework.get(mode_code, {})
                mode_name = mode_data.get('name', f"Mode {mode_code}")
                risk_score = float(mode_summary.loc[mode_code, 'transport_risk_score'])
                if risk_score >= 8:
                    priority = "CRITICAL"
                elif risk_score >= 6:
                    priority = "HIGH"
                else:
                    priority = "MEDIUM"

                print(f"{priority} PRIORITY: {mode_name} (Risk Score: {risk_score:.1f}/10)")
                if value_col:
                    total_value = float(mode_summary.loc[mode_code, value_col])
                    print(f"   • Value at risk: ${total_value/1e6:.1f}M")
                key_factors = mode_data.get('key_risk_factors', [])
                mitigations = mode_data.get('risk_mitigation', [])
                if key_factors:
                    print(f"   • Key risk factors: {', '.join(key_factors[:3])}")
                if mitigations:
                    print(f"   • Mitigation: {', '.join(mitigations[:2])}")
                print("")
    except Exception as exc:
        print(f"Note: Could not generate risk-based recommendations ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="International summary printer")
    parser.add_argument('--input', required=True, help='Path to pickled payload containing international_df (and optional ccc_analysis).')
    args = parser.parse_args()

    try:
        with open(args.input, 'rb') as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            payload = {'international_df': payload}
    except Exception as exc:
        print(f"Failed to load input payload: {exc}")
        sys.exit(1)

    try:
        build_summary(payload)
    except Exception as exc:
        print(f"Failed to generate summary: {exc}")
        sys.exit(2)


if __name__ == '__main__':
    main()
