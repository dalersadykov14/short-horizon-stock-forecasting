import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(train_prices: pd.Series, horizon: int = 30) -> float:
    """
    ARIMA forecast for log return at t+horizon.
    Fits ARIMA to log returns (computed with shift(1)).
    Returns the cumulative log return over the horizon.
    """
    if len(train_prices) < 120:
        raise ValueError("ARIMA needs at least 120 training points for stability.")

    prices = train_prices.astype(float)
    y = np.log(prices / prices.shift(1)).dropna()

    if len(y) < 60:
        raise ValueError("ARIMA needs at least 60 log return points for stability.")

    # Small grid to keep runtime sane + reduce blowups
    # You can expand later in Phase III.
    orders = [(0,0,0), (1,0,0), (0,0,1), (1,0,1), (2,0,0), (0,0,2), (2,0,1)]

    best = None
    best_aic = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in orders:
            try:
                model = ARIMA(y, order=order)
                fit = model.fit()
                aic = float(fit.aic)
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best = fit
            except Exception:
                continue

    if best is None:
        raise RuntimeError("ARIMA failed for all candidate orders.")

    # Forecast horizon steps; sum to get cumulative log return
    fc = best.forecast(steps=horizon)
    val = float(fc.sum())

    if not np.isfinite(val):
        raise ValueError("ARIMA produced non-finite forecast.")
    return val
