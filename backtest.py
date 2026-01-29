import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Dict, Any

def walk_forward_backtest(
    prices: pd.Series,
    horizon: int,
    min_train: int,
    step: int,
    model_fn: Callable[[pd.Series], float],
    model_name: str,
    ticker: str,
    log_fn: Callable[[str], None],
    target_mode: str = "price",
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    """
    Expanding window walk-forward backtest.
    Predicts terminal target at t+horizon.
    target_mode: "price" or "return" (log return).
    Returns preds, actuals, and a split log (with errors if any).
    """
    preds, actuals = [], []
    split_log = []

    n = len(prices)
    if n < (min_train + horizon + 5):
        raise ValueError(f"Not enough data for {ticker}: n={n}, need >= {min_train+horizon+5}")

    # split_end is the count of samples included in training
    # train_end_idx is the last index included in training
    split_end = min_train

    while (split_end - 1) + horizon < n:
        train = prices.iloc[:split_end]
        train_end_idx = split_end - 1
        target_idx = train_end_idx + horizon

        if target_idx <= train_end_idx:
            raise RuntimeError("Target index must be after train end (leakage check failed).")

        if target_mode == "price":
            target = float(prices.iloc[target_idx])
        elif target_mode == "return":
            base = float(prices.iloc[train_end_idx])
            future = float(prices.iloc[target_idx])
            target = float(np.log(future / base))
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

        record = {
            "ticker": ticker,
            "model": model_name,
            "train_end_date": str(train.index[-1].date()),
            "target_date": str(prices.index[target_idx].date()),
            "status": "ok",
            "error": "",
        }

        try:
            pred = float(model_fn(train))
            preds.append(pred)
            actuals.append(target)
        except Exception as e:
            record["status"] = "fail"
            record["error"] = repr(e)
            log_fn(f"[FAIL] {ticker} {model_name} train_end={record['train_end_date']} -> {record['error']}")

        split_log.append(record)
        split_end += step

    if len(actuals) == 0:
        raise RuntimeError(f"All splits failed or no splits executed for {ticker} / {model_name}")

    return preds, actuals, split_log
