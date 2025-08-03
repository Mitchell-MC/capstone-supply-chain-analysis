# Fix Corridor Analysis - Add this to your notebook

print("🛣️ TOP 10 INTERSTATE CORRIDORS (by volume):")
# Filter for interstate corridors only (different origin and destination)
interstate_corridors = df[df['dms_origst'] != df['dms_destst']]
top_interstate = interstate_corridors.groupby('corridor_names')['tons_2023'].sum().sort_values(ascending=False).head(10)

for corridor, volume in top_interstate.items():
    print(f"   {corridor:<35}: {volume/1e6:8.1f}M tons")

print(f"\n📊 INTERSTATE vs INTRSTATE BREAKDOWN:")
total_corridors = len(df)
intrastate_corridors = len(df[df['dms_origst'] == df['dms_destst']])
interstate_corridors = len(df[df['dms_origst'] != df['dms_destst']])

print(f"   Total corridors: {total_corridors:,}")
print(f"   Interstate corridors: {interstate_corridors:,} ({interstate_corridors/total_corridors*100:.1f}%)")
print(f"   Intrastate corridors: {intrastate_corridors:,} ({intrastate_corridors/total_corridors*100:.1f}%)")

# Show top interstate corridors by economic value
print(f"\n💰 TOP 10 INTERSTATE CORRIDORS (by economic value):")
top_value_corridors = interstate_corridors.groupby('corridor_names')['value_2023'].sum().sort_values(ascending=False).head(10)

for corridor, value in top_value_corridors.items():
    print(f"   {corridor:<35}: ${value/1e6:8.1f}M")

# Show regional interstate corridors
print(f"\n🗺️ TOP 10 REGIONAL INTERSTATE CORRIDORS (by volume):")
# Filter for corridors within same region
regional_corridors = interstate_corridors[interstate_corridors['origin_region'] == interstate_corridors['dest_region']]
top_regional = regional_corridors.groupby('corridor_names')['tons_2023'].sum().sort_values(ascending=False).head(10)

for corridor, volume in top_regional.items():
    print(f"   {corridor:<35}: {volume/1e6:8.1f}M tons")

# Show nearshoring opportunities (short distance interstate)
print(f"\n🎯 TOP 10 NEARSHORING OPPORTUNITIES (interstate, <250 miles):")
nearshore_interstate = interstate_corridors[interstate_corridors['dist_band'] <= 2]
top_nearshore = nearshore_interstate.groupby('corridor_names')['tons_2023'].sum().sort_values(ascending=False).head(10)

for corridor, volume in top_nearshore.items():
    print(f"   {corridor:<35}: {volume/1e6:8.1f}M tons") 