import numpy as np
import pandas as pd
from project_utils.metrics import create_percentile_score, compute_resilience_scores


def test_create_percentile_score_increasing():
    s = pd.Series([1, 2, 3, 4, 5])
    pct = create_percentile_score(s)
    assert pct.iloc[0] < pct.iloc[-1]


def test_create_percentile_score_invert():
    s = pd.Series([1, 2, 3, 4, 5])
    pct = create_percentile_score(s, invert=True)
    assert pct.iloc[0] > pct.iloc[-1]


def test_compute_resilience_scores_shapes():
    n = 100
    df = pd.DataFrame({
        'tons_2017': np.random.rand(n)*100+1,
        'tons_2023': np.random.rand(n)*100+1,
        'tons_volatility': np.random.rand(n),
        'corridor_concentration': np.random.rand(n)*100,
        'value_density': np.random.rand(n)*10,
    })
    scores = compute_resilience_scores(df, {'weights': {'stability':0.4,'growth':0.25,'diversification':0.25,'efficiency':0.1}})
    assert len(scores) == n
    assert float(scores.min()) >= 0
    assert float(scores.max()) <= 100
