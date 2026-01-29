import numpy as np
import pandas as pd

def naive_forecast(train_prices: pd.Series, horizon: int = 30) -> float:
    """Random walk baseline: expected log return over horizon is 0."""
    return 0.0

def drift_forecast(train_prices: pd.Series, horizon: int = 30) -> float:
    """Random walk with drift: mean past log return * horizon."""
    prices = train_prices.astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) == 0:
        raise ValueError("Not enough data for drift forecast.")
    mu = float(log_returns.mean())
    return mu * float(horizon)
