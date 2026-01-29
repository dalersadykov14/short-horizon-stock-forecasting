import numpy as np
import pandas as pd

def gbm_forecast(train_prices: pd.Series, horizon: int = 30, sims: int = 2000, seed: int = 42) -> float:
    """
    GBM forecast for log return at t+horizon.
    Uses log return mean/std from training window (computed with shift(1)).
    Returns expected log return over the horizon.
    """
    if len(train_prices) < 50:
        raise ValueError("GBM needs at least 50 training points.")

    prices = train_prices.astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 30:
        raise ValueError("GBM needs at least 30 log return points.")

    mu_daily = float(log_returns.mean())
    sigma_daily = float(log_returns.std(ddof=1))

    if sigma_daily <= 0 or not np.isfinite(sigma_daily):
        raise ValueError(f"GBM sigma invalid: {sigma_daily}")

    dt = 1.0  # 1 trading day steps (we simulate in daily units)

    rng = np.random.default_rng(seed)

    # simulate cumulative log returns over horizon
    shocks = rng.standard_normal(size=(sims, horizon))
    increments = mu_daily * dt + sigma_daily * np.sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    terminal_log_returns = log_paths[:, -1]

    if terminal_log_returns.size == 0 or not np.all(np.isfinite(terminal_log_returns)):
        raise ValueError("GBM produced non-finite terminal returns.")

    return float(np.mean(terminal_log_returns))
