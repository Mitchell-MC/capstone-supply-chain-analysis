# FAF5.7 Dataset Data Dictionary (Revised)
This document merges the auto-generated dictionary with authoritative metadata from `FAF5_metadata.xlsx`.

## Dataset Overview

- **Total Variables**: 56
- **Categories**: 6

## Variable Descriptions

| Column Name | Data Type | Category | Description | Units |
|---|---|---|---|---|
| fr_orig | float64 | Geographic | Foreign region of shipment origin Codes: 801=Canada; 802=Mexico; 803=Rest of Americas; 804=Europe; 805=Africa; 806=SW & Central Asia; 807=Eastern Asia; 808=SE Asia & Oceania | None |
| dms_origst | int64 | Geographic | Origin state FIPS code (1-56) Codes: 1=Alabama; 10=Delaware; 11=Washington DC; 12=Florida; 13=Georgia; 15=Hawaii; 16=Idaho; 17=Illinois; 18=Indiana; 19=Iowa; 2=Alaska; 20=Kansas; 21=Kentucky; 22=Louisiana; 23=Maine; 24=Maryland; 25=Massachusetts; 26=Michigan; 27=Minnesota; 28=Mississippi; 29=Missouri; 30=Montana; 31=Nebraska; 32=Nevada; 33=New Hampshire; 34=New Jersey; 35=New Mexico; 36=New York; 37=North Carolina; 38=North Dakota; 39=Ohio; 4=Arizona; 40=Oklahoma; 41=Oregon; 42=Pennsylvania; 44=Rhode Island; 45=South Carolina; 46=South Dakota; 47=Tennessee; 48=Texas; 49=Utah; 5=Arkansas; 50=Vermont; 51=Virginia; 53=Washington; 54=West Virginia; 55=Wisconsin; 56=Wyoming; 6=California; 8=Colorado; 9=Connecticut | nan |
| dms_destst | int64 | Geographic | Destination state FIPS code (1-56) Codes: 1=Alabama; 10=Delaware; 11=Washington DC; 12=Florida; 13=Georgia; 15=Hawaii; 16=Idaho; 17=Illinois; 18=Indiana; 19=Iowa; 2=Alaska; 20=Kansas; 21=Kentucky; 22=Louisiana; 23=Maine; 24=Maryland; 25=Massachusetts; 26=Michigan; 27=Minnesota; 28=Mississippi; 29=Missouri; 30=Montana; 31=Nebraska; 32=Nevada; 33=New Hampshire; 34=New Jersey; 35=New Mexico; 36=New York; 37=North Carolina; 38=North Dakota; 39=Ohio; 4=Arizona; 40=Oklahoma; 41=Oregon; 42=Pennsylvania; 44=Rhode Island; 45=South Carolina; 46=South Dakota; 47=Tennessee; 48=Texas; 49=Utah; 5=Arkansas; 50=Vermont; 51=Virginia; 53=Washington; 54=West Virginia; 55=Wisconsin; 56=Wyoming; 6=California; 8=Colorado; 9=Connecticut | nan |
| fr_dest | float64 | Geographic | Foreign region of shipment destination Codes: 801=Canada; 802=Mexico; 803=Rest of Americas; 804=Europe; 805=Africa; 806=SW & Central Asia; 807=Eastern Asia; 808=SE Asia & Oceania | None |
| fr_inmode | float64 | Transportation | Mode used between a foreign region and the US entry region for the imported goods Codes: 1=Road congestion, accidents, driver shortages, fuel price volatility, cargo theft. High flexibility but susceptible to last-mile delays and weather.; 2=Derailments, infrastructure bottlenecks, limited network flexibility, delays at intermodal hubs. Generally more reliable and secure for long-haul bulk than trucking.; 3=Extremely long and variable lead times, port congestion, customs delays, geopolitical chokepoints, cargo damage/loss at sea. Highly vulnerable to weather.; 4=Extremely high cost, capacity constraints, shipment handling damage, weather delays at airports. While fast, the financial and schedule risks are significant.; 5=Complexity at transfer points (intermodal risk), lack of single-point accountability, potential for delays at each hand-off. Risk is compounded.; 6=Catastrophic failure (leaks/spills), security threats, regulatory hurdles. Operationally very low-risk and reliable, but carries immense liability.; 7=Complete lack of visibility, control, and accountability. Represents a critical failure in the supply chain data stream, regardless of the specific cause.; 8=Concentrated dependency on a single point of entry. Vulnerable to localized disruptions like labor strikes, congestion, or regional weather events. | None |
| dms_mode | int64 | Transportation | Mode used between domestic origins and destinations Codes: 1=Road congestion, accidents, driver shortages, fuel price volatility, cargo theft. High flexibility but susceptible to last-mile delays and weather.; 2=Derailments, infrastructure bottlenecks, limited network flexibility, delays at intermodal hubs. Generally more reliable and secure for long-haul bulk than trucking.; 3=Extremely long and variable lead times, port congestion, customs delays, geopolitical chokepoints, cargo damage/loss at sea. Highly vulnerable to weather.; 4=Extremely high cost, capacity constraints, shipment handling damage, weather delays at airports. While fast, the financial and schedule risks are significant.; 5=Complexity at transfer points (intermodal risk), lack of single-point accountability, potential for delays at each hand-off. Risk is compounded.; 6=Catastrophic failure (leaks/spills), security threats, regulatory hurdles. Operationally very low-risk and reliable, but carries immense liability.; 7=Complete lack of visibility, control, and accountability. Represents a critical failure in the supply chain data stream, regardless of the specific cause.; 8=Concentrated dependency on a single point of entry. Vulnerable to localized disruptions like labor strikes, congestion, or regional weather events. | None |
| fr_outmode | float64 | Transportation | Mode used between the US exit region and foreign region for the exported goods Codes: 1=Road congestion, accidents, driver shortages, fuel price volatility, cargo theft. High flexibility but susceptible to last-mile delays and weather.; 2=Derailments, infrastructure bottlenecks, limited network flexibility, delays at intermodal hubs. Generally more reliable and secure for long-haul bulk than trucking.; 3=Extremely long and variable lead times, port congestion, customs delays, geopolitical chokepoints, cargo damage/loss at sea. Highly vulnerable to weather.; 4=Extremely high cost, capacity constraints, shipment handling damage, weather delays at airports. While fast, the financial and schedule risks are significant.; 5=Complexity at transfer points (intermodal risk), lack of single-point accountability, potential for delays at each hand-off. Risk is compounded.; 6=Catastrophic failure (leaks/spills), security threats, regulatory hurdles. Operationally very low-risk and reliable, but carries immense liability.; 7=Complete lack of visibility, control, and accountability. Represents a critical failure in the supply chain data stream, regardless of the specific cause.; 8=Concentrated dependency on a single point of entry. Vulnerable to localized disruptions like labor strikes, congestion, or regional weather events. | None |
| sctg2 | int64 | Commodity | 2-digit level of the Standard Classification of Transported Goods (SCTG) Codes: 1=Live animals/fish; 10=Building stone; 11=Natural sands; 12=Gravel; 13=Nonmetallic minerals; 14=Metallic ores; 15=Coal; 16=Crude petroleum; 17=Gasoline; 18=Fuel oils; 19=Natural gas and other fossil products; 2=Cereal grains; 20=Basic chemicals; 21=Pharmaceuticals; 22=Fertilizers; 23=Chemical prods.; 24=Plastics/rubber; 25=Logs; 26=Wood prods.; 27=Newsprint/paper; 28=Paper articles; 29=Printed prods.; 3=Other ag prods.; 30=Textiles/leather; 31=Nonmetal min. prods.; 32=Base metals; 33=Articles-base metal; 34=Machinery; 35=Electronics; 36=Motorized vehicles; 37=Transport equip.; 38=Precision instruments; 39=Furniture; 4=Animal feed; 40=Misc. mfg. prods.; 41=Waste/scrap; 43=Mixed freight; 5=Meat/seafood; 6=Milled grain prods.; 7=Other foodstuffs; 8=Alcoholic beverages; 9=Tobacco prods. | None |
| trade_type | int64 | Other | Type of trade (domestic, import, and export) Codes: 1=Domestic flows (freight shipments moved from US domestic origins to US domestic destinations); 2=Import flows (freight shipments moved from foreign origins to US domestic destinations); 3=Export flows (freight shipments moved from US domestic origins to foreign destinations) | None |
| dist_band | int64 | Other | Distance range for the average weighted distance of shipments. The distance miles were estimated for the US domestic portion only. For foreign trades, all “cutoff” locations are at the border or coastal zones, except for those with air modes. The “cutoff” location for air is the last airport where shipment leaving the U.S. for exports, or the first airport where shipment arriving the U.S. for imports.Codes: 1=Below 100; 2=100 - 249; 3=250 - 499; 4=500 - 749; 5=750 - 999; 6=1,000 - 1,499; 7=1,500 - 2,000; 8=Over 2,000 | None |
| tons_2017 | float64 | Tons_Data | Freight tons in 2017 | nan |
| tons_2018 | float64 | Tons_Data | Freight tons in 2018 | nan |
| tons_2019 | float64 | Tons_Data | Freight tons in 2019 | nan |
| tons_2020 | float64 | Tons_Data | Freight tons in 2020 | nan |
| tons_2021 | float64 | Tons_Data | Freight tons in 2021 | nan |
| tons_2022 | float64 | Tons_Data | Freight tons in 2022 | nan |
| tons_2023 | float64 | Tons_Data | Freight tons in 2023 | nan |
| tons_2024 | float64 | Tons_Data | Projected freight tons in 2024 | nan |
| tons_2030 | float64 | Tons_Data | Projected freight tons in 2030 | nan |
| tons_2035 | float64 | Tons_Data | Projected freight tons in 2035 | nan |
| tons_2040 | float64 | Tons_Data | Projected freight tons in 2040 | nan |
| tons_2045 | float64 | Tons_Data | Projected freight tons in 2045 | nan |
| tons_2050 | float64 | Tons_Data | Projected freight tons in 2050 | nan |
| value_2017 | float64 | Value_Data | Freight value in 2017 (thousands of dollars) | nan |
| value_2018 | float64 | Value_Data | Freight value in 2018 (thousands of dollars) | nan |
| value_2019 | float64 | Value_Data | Freight value in 2019 (thousands of dollars) | nan |
| value_2020 | float64 | Value_Data | Freight value in 2020 (thousands of dollars) | nan |
| value_2021 | float64 | Value_Data | Freight value in 2021 (thousands of dollars) | nan |
| value_2022 | float64 | Value_Data | Freight value in 2022 (thousands of dollars) | nan |
| value_2023 | float64 | Value_Data | Freight value in 2023 (thousands of dollars) | nan |
| value_2024 | float64 | Value_Data | Projected freight value in 2024 (thousands of dollars) | nan |
| value_2030 | float64 | Value_Data | No description available | nan |
| value_2035 | float64 | Value_Data | No description available | nan |
| value_2040 | float64 | Value_Data | No description available | nan |
| value_2045 | float64 | Value_Data | No description available | nan |
| value_2050 | float64 | Value_Data | No description available | nan |
| current_value_2018 | float64 | Other | No description available | nan |
| current_value_2019 | float64 | Other | No description available | nan |
| current_value_2020 | float64 | Other | No description available | nan |
| current_value_2021 | float64 | Other | No description available | nan |
| current_value_2022 | float64 | Other | No description available | nan |
| current_value_2023 | float64 | Other | No description available | nan |
| current_value_2024 | float64 | Other | No description available | nan |
| tmiles_2017 | float64 | Other | No description available | nan |
| tmiles_2018 | float64 | Other | No description available | nan |
| tmiles_2019 | float64 | Other | No description available | nan |
| tmiles_2020 | float64 | Other | No description available | nan |
| tmiles_2021 | float64 | Other | No description available | nan |
| tmiles_2022 | float64 | Other | No description available | nan |
| tmiles_2023 | float64 | Other | Transport miles in 2023 | nan |
| tmiles_2024 | float64 | Other | No description available | nan |
| tmiles_2030 | float64 | Other | No description available | nan |
| tmiles_2035 | float64 | Other | No description available | nan |
| tmiles_2040 | float64 | Other | No description available | nan |
| tmiles_2045 | float64 | Other | No description available | nan |
| tmiles_2050 | float64 | Other | No description available | nan |
