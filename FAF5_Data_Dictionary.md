# FAF5.7 Dataset Data Dictionary

This document provides a comprehensive description of all variables in the FAF5.7 freight dataset.

## Dataset Overview

- **Total Records**: 56
- **Total Variables**: 56
- **Generated**: 2025-08-10 10:37:26

## Variable Descriptions

| Column Name | Data Type | Description | Missing % | Category |
|-------------|-----------|-------------|-----------|----------|
| fr_orig | float64 | Foreign origin region code (801-808 for international) | 57.1% | Geographic |
| dms_origst | int64 | Origin state FIPS code (1-56) | 0.0% | Geographic |
| dms_destst | int64 | Destination state FIPS code (1-56) | 0.0% | Geographic |
| fr_dest | float64 | Foreign destination region code (801-808 for international) | 58.1% | Geographic |
| fr_inmode | float64 | No description available | 57.1% | Transportation |
| dms_mode | int64 | Transport mode (1=Truck, 2=Rail, 3=Water, 4=Air, 5=Pipeline) | 0.0% | Transportation |
| fr_outmode | float64 | No description available | 58.1% | Transportation |
| sctg2 | int64 | Standard Classification of Transported Goods (2-digit) | 0.0% | Commodity |
| trade_type | int64 | Trade type (1=Domestic, 2=Import, 3=Export) | 0.0% | Other |
| dist_band | int64 | Distance band classification | 0.0% | Other |
| tons_2017 | float64 | Freight tons in 2017 | 0.0% | Tons_Data |
| tons_2018 | float64 | Freight tons in 2018 | 0.0% | Tons_Data |
| tons_2019 | float64 | Freight tons in 2019 | 0.0% | Tons_Data |
| tons_2020 | float64 | Freight tons in 2020 | 0.0% | Tons_Data |
| tons_2021 | float64 | Freight tons in 2021 | 0.0% | Tons_Data |
| tons_2022 | float64 | Freight tons in 2022 | 0.0% | Tons_Data |
| tons_2023 | float64 | Freight tons in 2023 | 0.0% | Tons_Data |
| tons_2024 | float64 | Projected freight tons in 2024 | 0.0% | Tons_Data |
| tons_2030 | float64 | Projected freight tons in 2030 | 0.0% | Tons_Data |
| tons_2035 | float64 | Projected freight tons in 2035 | 0.0% | Tons_Data |
| tons_2040 | float64 | Projected freight tons in 2040 | 0.0% | Tons_Data |
| tons_2045 | float64 | Projected freight tons in 2045 | 0.0% | Tons_Data |
| tons_2050 | float64 | Projected freight tons in 2050 | 0.0% | Tons_Data |
| value_2017 | float64 | Freight value in 2017 (thousands of dollars) | 0.0% | Value_Data |
| value_2018 | float64 | Freight value in 2018 (thousands of dollars) | 0.0% | Value_Data |
| value_2019 | float64 | Freight value in 2019 (thousands of dollars) | 0.0% | Value_Data |
| value_2020 | float64 | Freight value in 2020 (thousands of dollars) | 0.0% | Value_Data |
| value_2021 | float64 | Freight value in 2021 (thousands of dollars) | 0.0% | Value_Data |
| value_2022 | float64 | Freight value in 2022 (thousands of dollars) | 0.0% | Value_Data |
| value_2023 | float64 | Freight value in 2023 (thousands of dollars) | 0.0% | Value_Data |
| value_2024 | float64 | Projected freight value in 2024 (thousands of dollars) | 0.0% | Value_Data |
| value_2030 | float64 | No description available | 0.0% | Value_Data |
| value_2035 | float64 | No description available | 0.0% | Value_Data |
| value_2040 | float64 | No description available | 0.0% | Value_Data |
| value_2045 | float64 | No description available | 0.0% | Value_Data |
| value_2050 | float64 | No description available | 0.0% | Value_Data |
| current_value_2018 | float64 | No description available | 0.0% | Other |
| current_value_2019 | float64 | No description available | 0.0% | Other |
| current_value_2020 | float64 | No description available | 0.0% | Other |
| current_value_2021 | float64 | No description available | 0.0% | Other |
| current_value_2022 | float64 | No description available | 0.0% | Other |
| current_value_2023 | float64 | No description available | 0.0% | Other |
| current_value_2024 | float64 | No description available | 0.0% | Other |
| tmiles_2017 | float64 | No description available | 0.0% | Other |
| tmiles_2018 | float64 | No description available | 0.0% | Other |
| tmiles_2019 | float64 | No description available | 0.0% | Other |
| tmiles_2020 | float64 | No description available | 0.0% | Other |
| tmiles_2021 | float64 | No description available | 0.0% | Other |
| tmiles_2022 | float64 | No description available | 0.0% | Other |
| tmiles_2023 | float64 | Transport miles in 2023 | 0.0% | Other |
| tmiles_2024 | float64 | No description available | 0.0% | Other |
| tmiles_2030 | float64 | No description available | 0.0% | Other |
| tmiles_2035 | float64 | No description available | 0.0% | Other |
| tmiles_2040 | float64 | No description available | 0.0% | Other |
| tmiles_2045 | float64 | No description available | 0.0% | Other |
| tmiles_2050 | float64 | No description available | 0.0% | Other |
