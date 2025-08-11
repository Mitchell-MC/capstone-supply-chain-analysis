from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def create_percentile_score(series: pd.Series, invert: bool = False) -> pd.Series:
    series_clean = series.replace([np.inf, -np.inf], np.nan).fillna(series.median())
    pct = series_clean.rank(pct=True) * 100.0
    return 100.0 - pct if invert else pct


def compute_resilience_scores(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    required = {'tons_2017','tons_2023'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stability = create_percentile_score(df['tons_volatility'], invert=True) if 'tons_volatility' in df.columns else pd.Series(50.0, index=df.index)
    growth_raw = (df['tons_2023'] - df['tons_2017']) / (df['tons_2017'] + 1e-3)
    qlo, qhi = growth_raw.quantile(0.05), growth_raw.quantile(0.95)
    growth = create_percentile_score(growth_raw.clip(qlo, qhi))

    diversification = create_percentile_score(df['corridor_concentration'], invert=True) if 'corridor_concentration' in df.columns else pd.Series(50.0, index=df.index)
    efficiency = create_percentile_score(df['value_density']) if 'value_density' in df.columns else pd.Series(50.0, index=df.index)

    w = cfg.get('weights', {'stability':0.4,'growth':0.25,'diversification':0.25,'efficiency':0.1})
    return (
        stability * w['stability'] +
        growth * w['growth'] +
        diversification * w['diversification'] +
        efficiency * w['efficiency']
    )
