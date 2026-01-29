import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

from data import load_prices
from backtest import walk_forward_backtest
from models_gbm import gbm_forecast
from models_arima import arima_forecast
from models_naive import naive_forecast, drift_forecast
from metrics import (
    rmse,
    mae,
    nrmse,
    nmae,
    directional_accuracy_neutral,
    balanced_accuracy_posneg,
    sign_audit,
    mase,
    rmsse,
)
from plotting import save_pred_vs_actual_plot, save_error_hist_plot

TICKERS = ["SPY", "VOO", "MSFT", "AAPL"]
HORIZON = 30          # trading days
MIN_TRAIN = 252       # ~1 year
STEP = 30             # advance by horizon (less overlap)
GBM_SIMS = 2000
SEED = 42
TARGET_MODE = "return"
EPS_FLOOR = 0.005
K_BASE = 1.0
VOL_WINDOW = 60
SMA_WINDOW = 200
WEIGHT_TAU = 0.02

def ensure_dirs():
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

def log_line(msg: str, logfile: str):
    print(msg)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def write_results(rows, out_csv):
    if not rows:
        return None
    results = pd.DataFrame(rows).sort_values(["ticker", "model"])
    results.to_csv(out_csv, index=False)
    return results

def append_sign_audit_rows(rows, ticker, model, preds, actuals, eps):
    if preds is None or actuals is None:
        return

    n = min(len(preds), len(actuals))
    if n == 0:
        return

    def add_segment(segment_name, start, end):
        seg_preds = preds[start:end]
        seg_actuals = actuals[start:end]
        audit = sign_audit(seg_preds, eps)
        rows.append({
            "ticker": ticker,
            "model": model,
            "segment": segment_name,
            "pos_rate": audit["pos_rate"],
            "neg_rate": audit["neg_rate"],
            "neutral_rate": audit["neutral_rate"],
            "coverage": audit["coverage"],
            "mean_pred": audit["mean_pred"],
            "std_pred": audit["std_pred"],
            "DA": directional_accuracy_neutral(seg_actuals, seg_preds, eps),
            "BA": balanced_accuracy_posneg(seg_actuals, seg_preds, eps),
            "eps": eps,
        })

    add_segment("all", 0, n)

    third = max(1, n // 3)
    add_segment("early", 0, third)
    add_segment("mid", third, min(2 * third, n))
    add_segment("late", min(2 * third, n), n)

def horizon_log_returns(prices: pd.Series, horizon: int) -> pd.Series:
    vals = prices.astype(float)
    return np.log(vals.shift(-horizon) / vals).dropna()

def epsilon_from_train(r30_train: pd.Series) -> float:
    if r30_train is None or len(r30_train) == 0:
        return EPS_FLOOR
    sigma = float(np.std(r30_train, ddof=1)) if len(r30_train) > 1 else 0.0
    return float(max(EPS_FLOOR, 0.10 * sigma))

def skill_mae(y_true, y_pred, baseline_pred) -> float:
    base = mae(y_true, baseline_pred)
    return 1.0 - (mae(y_true, y_pred) / base) if base != 0 else float("nan")

def skill_mse(y_true, y_pred, baseline_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    baseline_pred = np.array(baseline_pred, dtype=float)
    base = float(np.mean((y_true - baseline_pred) ** 2)) if len(y_true) > 0 else float("nan")
    model = float(np.mean((y_true - y_pred) ** 2)) if len(y_true) > 0 else float("nan")
    return 1.0 - (model / base) if base != 0 and not np.isnan(base) else float("nan")

def append_errors(errors_csv, ticker, model, split_log, preds, actuals):
    if not split_log or not preds or not actuals:
        return

    os.makedirs(os.path.dirname(errors_csv), exist_ok=True)

    rows = []
    pred_idx = 0
    for split_idx, rec in enumerate(split_log):
        if rec.get("status") != "ok":
            continue
        if pred_idx >= len(preds) or pred_idx >= len(actuals):
            break
        pred = float(preds[pred_idx])
        actual = float(actuals[pred_idx])
        rows.append({
            "ticker": ticker,
            "model": model,
            "split_index": split_idx,
            "train_end_date": rec.get("train_end_date", ""),
            "target_date": rec.get("target_date", ""),
            "prediction": pred,
            "actual": actual,
            "error": pred - actual,
        })
        pred_idx += 1

    if not rows:
        return

    write_header = not os.path.exists(errors_csv)
    pd.DataFrame(rows).to_csv(errors_csv, mode="a", index=False, header=write_header)

def build_split_map(split_log, preds, actuals, prices, horizon, min_train, step):
    split_map = {}
    pred_idx = 0
    for split_idx, rec in enumerate(split_log):
        if rec.get("status") != "ok":
            continue
        if pred_idx >= len(preds) or pred_idx >= len(actuals):
            break
        train_end_idx = min_train - 1 + split_idx * step
        target_idx = train_end_idx + horizon
        split_map[split_idx] = {
            "prediction": float(preds[pred_idx]),
            "actual": float(actuals[pred_idx]),
            "train_end_date": rec.get("train_end_date", ""),
            "target_date": rec.get("target_date", ""),
            "train_end_idx": train_end_idx,
            "target_idx": target_idx,
        }
        pred_idx += 1
    return split_map

def compute_weights_from_history(error_history, tau, min_weight=0.10):
    models = list(error_history.keys())
    maes = {}
    for m in models:
        errs = error_history[m]
        maes[m] = float(np.mean(errs)) if errs else None

    if any(maes[m] is None for m in models):
        weights = {m: 1.0 / len(models) for m in models}
    else:
        scores = {m: float(np.exp(-maes[m] / tau)) for m in models}
        total = sum(scores.values())
        weights = {m: (scores[m] / total if total > 0 else 0.0) for m in models}

    for m in weights:
        weights[m] = max(weights[m], min_weight)
    norm = sum(weights.values())
    if norm > 0:
        weights = {m: w / norm for m, w in weights.items()}

    return weights, maes

def ensemble_from_maps(split_maps, prices, horizon, vol_window, sma_window, min_train, step, tau):
    common = None
    for smap in split_maps.values():
        keys = set(smap.keys())
        common = keys if common is None else common.intersection(keys)

    if not common:
        return [], [], [], [], [], [], []

    daily_log_returns = np.log(prices / prices.shift(1))
    rolling_vol = daily_log_returns.rolling(vol_window).std(ddof=1)
    sma = prices.rolling(sma_window).mean()

    models = list(split_maps.keys())
    error_history = {m: [] for m in models}
    preds = []
    actuals = []
    log = []
    signals = []
    sigmas = []
    regimes = []
    weights_out = []

    for idx in sorted(common):
        weights, maes = compute_weights_from_history(error_history, tau)
        weights_out.append({"split_index": idx, **weights})

        pred = 0.0
        for m, w in weights.items():
            pred += w * split_maps[m][idx]["prediction"]

        actual = split_maps[models[0]][idx]["actual"]
        preds.append(float(pred))
        actuals.append(float(actual))

        train_end_idx = split_maps[models[0]][idx]["train_end_idx"]
        sigma = float(rolling_vol.iloc[train_end_idx]) * np.sqrt(horizon) if train_end_idx >= 0 else float("nan")
        sigma = sigma if np.isfinite(sigma) and sigma > 0 else float("nan")
        sigmas.append(sigma)

        price_val = float(prices.iloc[train_end_idx])
        sma_val = float(sma.iloc[train_end_idx]) if np.isfinite(sma.iloc[train_end_idx]) else float("nan")
        regime = bool(price_val > sma_val) if np.isfinite(sma_val) else False
        regimes.append(regime)

        signal = float(pred / sigma) if sigma is not None and np.isfinite(sigma) and sigma > 0 else 0.0
        signals.append(signal)

        log.append({
            "split_index": idx,
            "train_end_date": split_maps[models[0]][idx]["train_end_date"],
            "target_date": split_maps[models[0]][idx]["target_date"],
            "status": "ok",
        })

        for m in models:
            error_history[m].append(abs(split_maps[m][idx]["prediction"] - actual))

    return preds, actuals, log, signals, sigmas, regimes, weights_out

def k_sweep_metrics(actuals, signals, regimes, base_k, multipliers):
    rows = []
    for mult in multipliers:
        k = float(base_k * mult)
        trade_mask = (np.array(signals) > k) & np.array(regimes, dtype=bool)
        coverage = float(np.mean(trade_mask)) if len(signals) > 0 else float("nan")
        avg_signed_return = float(np.mean(np.array(actuals)[trade_mask])) if np.any(trade_mask) else float("nan")
        da = float(np.mean(np.array(actuals)[trade_mask] > 0)) if np.any(trade_mask) else float("nan")
        rows.append({
            "k_mult": mult,
            "k": k,
            "coverage": coverage,
            "da": da,
            "avg_signed_return": avg_signed_return,
        })
    return rows

def trade_simulation(actuals, signals, sigmas, regimes, k, horizon, cost_bps=5, target_vol=0.10, min_hold=1):
    if len(actuals) == 0:
        return {
            "trades": 0,
            "coverage": float("nan"),
            "total_return": float("nan"),
            "cagr": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "turnover": float("nan"),
        }

    cost = cost_bps / 10000.0
    equity = 1.0
    peak = 1.0
    drawdowns = []
    pnl_list = []
    positions = []
    hold = 0
    pos = 0.0

    for signal, actual, sigma, regime in zip(signals, actuals, sigmas, regimes):
        desired = 0.0
        if regime and signal > k:
            desired = target_vol / sigma if sigma and np.isfinite(sigma) else 0.0
            desired = float(min(1.0, max(0.0, desired)))

        if pos > 0:
            hold += 1
            if (not regime or signal < 0) and hold >= min_hold:
                desired = 0.0
        else:
            if desired > 0:
                hold = 0

        trade_cost = cost * abs(desired - pos)
        pos = desired
        positions.append(pos)
        pnl = pos * actual - trade_cost
        pnl_list.append(pnl)
        equity *= (1.0 + pnl)
        peak = max(peak, equity)
        drawdowns.append((equity / peak) - 1.0)

    pnl_arr = np.array(pnl_list, dtype=float)
    hits = pnl_arr[pnl_arr != 0]
    hit_rate = float(np.mean(hits > 0)) if hits.size > 0 else float("nan")
    avg_win = float(np.mean(hits[hits > 0])) if np.any(hits > 0) else float("nan")
    avg_loss = float(np.mean(hits[hits < 0])) if np.any(hits < 0) else float("nan")

    periods = len(actuals)
    ann_factor = 252.0 / float(horizon)
    cagr = float(equity ** (ann_factor / periods) - 1.0) if periods > 0 else float("nan")
    sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr, ddof=1) * np.sqrt(ann_factor)) if pnl_arr.size > 1 and np.std(pnl_arr, ddof=1) > 0 else float("nan")
    years = periods / ann_factor if ann_factor > 0 else float("nan")
    trades_per_year = float(np.sum(np.abs(np.diff(positions)) > 0) / years) if years and years > 0 else float("nan")

    return {
        "trades": int(np.sum(np.array(positions) != 0)),
        "coverage": float(np.mean(np.array(positions) != 0)),
        "total_return": float(equity - 1.0),
        "cagr": cagr,
        "max_drawdown": float(min(drawdowns)) if drawdowns else float("nan"),
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "turnover": float(np.mean(np.abs(np.diff(positions)))) if len(positions) > 1 else float("nan"),
        "trades_per_year": trades_per_year,
    }

def main():
    ensure_dirs()
    np.random.seed(SEED)
    warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"outputs/logs/run_{run_id}.log"
    log_jsonl = f"outputs/logs/splits_{run_id}.jsonl"

    rows = []
    sign_audit_rows = []
    out_csv = f"outputs/tables/results_{run_id}.csv"
    errors_csv = f"outputs/tables/errors_{run_id}.csv"
    sign_csv = f"outputs/tables/sign_audit_{run_id}.csv"
    weights_csv = f"outputs/tables/weights_{run_id}.csv"
    k_csv = f"outputs/tables/k_sweep_{run_id}.csv"
    trade_csv = f"outputs/tables/trade_sim_{run_id}.csv"

    results = None
    weights_rows = []
    k_rows = []
    trade_rows = []

    try:
        for ticker in TICKERS:
            log_line(f"[START] {ticker}", logfile)
            try:
                prices = load_prices(ticker, years=5)
            except Exception as e:
                log_line(f"[SKIP] {ticker}: data error -> {repr(e)}", logfile)
                continue

            if len(prices) < (MIN_TRAIN + HORIZON + 10):
                log_line(f"[SKIP] {ticker}: not enough data (n={len(prices)})", logfile)
                continue

            gbm_preds, gbm_actuals, gbm_log = [], [], []
            arima_preds, arima_actuals, arima_log = [], [], []
            naive_preds, naive_actuals, naive_log = [], [], []
            drift_preds, drift_actuals, drift_log = [], [], []

            r30_train = horizon_log_returns(prices.iloc[:MIN_TRAIN], HORIZON)
            eps = epsilon_from_train(r30_train)

            # --- DRIFT ---
            try:
                drift_fn = lambda train: drift_forecast(train, horizon=HORIZON)
                drift_preds, drift_actuals, drift_log = walk_forward_backtest(
                    prices=prices,
                    horizon=HORIZON,
                    min_train=MIN_TRAIN,
                    step=STEP,
                    model_fn=drift_fn,
                    model_name="DRIFT",
                    ticker=ticker,
                    log_fn=lambda m: log_line(m, logfile),
                    target_mode=TARGET_MODE,
                )

                save_pred_vs_actual_plot(ticker, "DRIFT", drift_preds, drift_actuals, target_label="Return")
                save_error_hist_plot(ticker, "DRIFT", drift_preds, drift_actuals, target_label="Return")

                scale_train = r30_train
                baseline_zero = np.zeros(len(drift_actuals), dtype=float)

                rows.append({
                    "ticker": ticker,
                    "model": "DRIFT",
                    "horizon": HORIZON,
                    "n_splits_total": len(drift_log),
                    "n_splits_ok": len(drift_actuals),
                    "RMSE": rmse(drift_actuals, drift_preds),
                    "MAE": mae(drift_actuals, drift_preds),
                    "nRMSE": nrmse(drift_actuals, drift_preds),
                    "nMAE": nmae(drift_actuals, drift_preds),
                    "DA": directional_accuracy_neutral(drift_actuals, drift_preds, eps),
                    "Coverage": sign_audit(drift_preds, eps)["coverage"],
                    "BA": balanced_accuracy_posneg(drift_actuals, drift_preds, eps),
                    "MASE": mase(drift_actuals, drift_preds, scale_train),
                    "RMSSE": rmsse(drift_actuals, drift_preds, scale_train),
                    "Skill_MAE_0": skill_mae(drift_actuals, drift_preds, baseline_zero),
                    "Skill_MSE_0": skill_mse(drift_actuals, drift_preds, baseline_zero),
                    "Skill_MAE_Drift": 0.0,
                    "Skill_MSE_Drift": 0.0,
                    "Eps": eps,
                })

                append_errors(errors_csv, ticker, "DRIFT", drift_log, drift_preds, drift_actuals)
                append_sign_audit_rows(sign_audit_rows, ticker, "DRIFT", drift_preds, drift_actuals, eps)
            except Exception as e:
                log_line(f"[FAIL] {ticker} DRIFT -> {repr(e)}", logfile)

            # --- GBM ---
            try:
                gbm_fn = lambda train: gbm_forecast(train, horizon=HORIZON, sims=GBM_SIMS, seed=SEED)
                gbm_preds, gbm_actuals, gbm_log = walk_forward_backtest(
                    prices=prices,
                    horizon=HORIZON,
                    min_train=MIN_TRAIN,
                    step=STEP,
                    model_fn=gbm_fn,
                    model_name="GBM",
                    ticker=ticker,
                    log_fn=lambda m: log_line(m, logfile),
                    target_mode=TARGET_MODE,
                )

                save_pred_vs_actual_plot(ticker, "GBM", gbm_preds, gbm_actuals, target_label="Return")
                save_error_hist_plot(ticker, "GBM", gbm_preds, gbm_actuals, target_label="Return")

                scale_train = r30_train
                baseline_zero = np.zeros(len(gbm_actuals), dtype=float)
                baseline_drift = drift_preds if len(drift_preds) == len(gbm_actuals) else None

                rows.append({
                    "ticker": ticker,
                    "model": "GBM",
                    "horizon": HORIZON,
                    "n_splits_total": len(gbm_log),
                    "n_splits_ok": len(gbm_actuals),
                    "RMSE": rmse(gbm_actuals, gbm_preds),
                    "MAE": mae(gbm_actuals, gbm_preds),
                    "nRMSE": nrmse(gbm_actuals, gbm_preds),
                    "nMAE": nmae(gbm_actuals, gbm_preds),
                    "DA": directional_accuracy_neutral(gbm_actuals, gbm_preds, eps),
                    "Coverage": sign_audit(gbm_preds, eps)["coverage"],
                    "BA": balanced_accuracy_posneg(gbm_actuals, gbm_preds, eps),
                    "MASE": mase(gbm_actuals, gbm_preds, scale_train),
                    "RMSSE": rmsse(gbm_actuals, gbm_preds, scale_train),
                    "Skill_MAE_0": skill_mae(gbm_actuals, gbm_preds, baseline_zero),
                    "Skill_MSE_0": skill_mse(gbm_actuals, gbm_preds, baseline_zero),
                    "Skill_MAE_Drift": skill_mae(gbm_actuals, gbm_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                    "Skill_MSE_Drift": skill_mse(gbm_actuals, gbm_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                    "Eps": eps,
                })

                append_errors(errors_csv, ticker, "GBM", gbm_log, gbm_preds, gbm_actuals)
                append_sign_audit_rows(sign_audit_rows, ticker, "GBM", gbm_preds, gbm_actuals, eps)
            except Exception as e:
                log_line(f"[FAIL] {ticker} GBM -> {repr(e)}", logfile)

            # --- ARIMA ---
            try:
                arima_fn = lambda train: arima_forecast(train, horizon=HORIZON)
                arima_preds, arima_actuals, arima_log = walk_forward_backtest(
                    prices=prices,
                    horizon=HORIZON,
                    min_train=MIN_TRAIN,
                    step=STEP,
                    model_fn=arima_fn,
                    model_name="ARIMA",
                    ticker=ticker,
                    log_fn=lambda m: log_line(m, logfile),
                    target_mode=TARGET_MODE,
                )

                save_pred_vs_actual_plot(ticker, "ARIMA", arima_preds, arima_actuals, target_label="Return")
                save_error_hist_plot(ticker, "ARIMA", arima_preds, arima_actuals, target_label="Return")

                scale_train = r30_train
                baseline_zero = np.zeros(len(arima_actuals), dtype=float)
                baseline_drift = drift_preds if len(drift_preds) == len(arima_actuals) else None

                rows.append({
                    "ticker": ticker,
                    "model": "ARIMA",
                    "horizon": HORIZON,
                    "n_splits_total": len(arima_log),
                    "n_splits_ok": len(arima_actuals),
                    "RMSE": rmse(arima_actuals, arima_preds),
                    "MAE": mae(arima_actuals, arima_preds),
                    "nRMSE": nrmse(arima_actuals, arima_preds),
                    "nMAE": nmae(arima_actuals, arima_preds),
                    "DA": directional_accuracy_neutral(arima_actuals, arima_preds, eps),
                    "Coverage": sign_audit(arima_preds, eps)["coverage"],
                    "BA": balanced_accuracy_posneg(arima_actuals, arima_preds, eps),
                    "MASE": mase(arima_actuals, arima_preds, scale_train),
                    "RMSSE": rmsse(arima_actuals, arima_preds, scale_train),
                    "Skill_MAE_0": skill_mae(arima_actuals, arima_preds, baseline_zero),
                    "Skill_MSE_0": skill_mse(arima_actuals, arima_preds, baseline_zero),
                    "Skill_MAE_Drift": skill_mae(arima_actuals, arima_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                    "Skill_MSE_Drift": skill_mse(arima_actuals, arima_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                    "Eps": eps,
                })

                append_errors(errors_csv, ticker, "ARIMA", arima_log, arima_preds, arima_actuals)
                append_sign_audit_rows(sign_audit_rows, ticker, "ARIMA", arima_preds, arima_actuals, eps)
            except Exception as e:
                log_line(f"[FAIL] {ticker} ARIMA -> {repr(e)}", logfile)

            # --- NAIVE ---
            try:
                naive_fn = lambda train: naive_forecast(train, horizon=HORIZON)
                naive_preds, naive_actuals, naive_log = walk_forward_backtest(
                    prices=prices,
                    horizon=HORIZON,
                    min_train=MIN_TRAIN,
                    step=STEP,
                    model_fn=naive_fn,
                    model_name="NAIVE",
                    ticker=ticker,
                    log_fn=lambda m: log_line(m, logfile),
                    target_mode=TARGET_MODE,
                )

                save_pred_vs_actual_plot(ticker, "NAIVE", naive_preds, naive_actuals, target_label="Return")
                save_error_hist_plot(ticker, "NAIVE", naive_preds, naive_actuals, target_label="Return")

                scale_train = r30_train
                baseline_zero = np.zeros(len(naive_actuals), dtype=float)
                baseline_drift = drift_preds if len(drift_preds) == len(naive_actuals) else None

                rows.append({
                    "ticker": ticker,
                    "model": "NAIVE",
                    "horizon": HORIZON,
                    "n_splits_total": len(naive_log),
                    "n_splits_ok": len(naive_actuals),
                    "RMSE": rmse(naive_actuals, naive_preds),
                    "MAE": mae(naive_actuals, naive_preds),
                    "nRMSE": nrmse(naive_actuals, naive_preds),
                    "nMAE": nmae(naive_actuals, naive_preds),
                    "DA": directional_accuracy_neutral(naive_actuals, naive_preds, eps),
                    "Coverage": sign_audit(naive_preds, eps)["coverage"],
                    "BA": balanced_accuracy_posneg(naive_actuals, naive_preds, eps),
                    "MASE": mase(naive_actuals, naive_preds, scale_train),
                    "RMSSE": rmsse(naive_actuals, naive_preds, scale_train),
                    "Skill_MAE_0": skill_mae(naive_actuals, naive_preds, baseline_zero),
                    "Skill_MSE_0": skill_mse(naive_actuals, naive_preds, baseline_zero),
                    "Skill_MAE_Drift": skill_mae(naive_actuals, naive_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                    "Skill_MSE_Drift": skill_mse(naive_actuals, naive_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                    "Eps": eps,
                })

                append_errors(errors_csv, ticker, "NAIVE", naive_log, naive_preds, naive_actuals)
                append_sign_audit_rows(sign_audit_rows, ticker, "NAIVE", naive_preds, naive_actuals, eps)
            except Exception as e:
                log_line(f"[FAIL] {ticker} NAIVE -> {repr(e)}", logfile)

            # --- ENSEMBLE ---
            try:
                model_maps = {
                    "DRIFT": build_split_map(drift_log, drift_preds, drift_actuals, prices, HORIZON, MIN_TRAIN, STEP),
                    "GBM": build_split_map(gbm_log, gbm_preds, gbm_actuals, prices, HORIZON, MIN_TRAIN, STEP),
                    "ARIMA": build_split_map(arima_log, arima_preds, arima_actuals, prices, HORIZON, MIN_TRAIN, STEP),
                }

                ens_preds, ens_actuals, ens_log, ens_signals, ens_sigmas, ens_regimes, ens_weights = ensemble_from_maps(
                    model_maps,
                    prices,
                    HORIZON,
                    VOL_WINDOW,
                    SMA_WINDOW,
                    MIN_TRAIN,
                    STEP,
                    WEIGHT_TAU,
                )

                if ens_preds and ens_actuals:
                    save_pred_vs_actual_plot(ticker, "ENSEMBLE", ens_preds, ens_actuals, target_label="Return")
                    save_error_hist_plot(ticker, "ENSEMBLE", ens_preds, ens_actuals, target_label="Return")

                    scale_train = r30_train
                    baseline_zero = np.zeros(len(ens_actuals), dtype=float)
                    baseline_drift = drift_preds if len(drift_preds) == len(ens_actuals) else None

                    rows.append({
                        "ticker": ticker,
                        "model": "ENSEMBLE",
                        "horizon": HORIZON,
                        "n_splits_total": len(ens_log),
                        "n_splits_ok": len(ens_actuals),
                        "RMSE": rmse(ens_actuals, ens_preds),
                        "MAE": mae(ens_actuals, ens_preds),
                        "nRMSE": nrmse(ens_actuals, ens_preds),
                        "nMAE": nmae(ens_actuals, ens_preds),
                        "DA": float(np.mean((np.array(ens_actuals) > 0) & (np.array(ens_signals) > K_BASE) & np.array(ens_regimes, dtype=bool))) if ens_actuals else float("nan"),
                        "Coverage": float(np.mean((np.array(ens_signals) > K_BASE) & np.array(ens_regimes, dtype=bool))) if ens_actuals else float("nan"),
                        "BA": float("nan"),
                        "MASE": mase(ens_actuals, ens_preds, scale_train),
                        "RMSSE": rmsse(ens_actuals, ens_preds, scale_train),
                        "Skill_MAE_0": skill_mae(ens_actuals, ens_preds, baseline_zero),
                        "Skill_MSE_0": skill_mse(ens_actuals, ens_preds, baseline_zero),
                        "Skill_MAE_Drift": skill_mae(ens_actuals, ens_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                        "Skill_MSE_Drift": skill_mse(ens_actuals, ens_preds, baseline_drift) if baseline_drift is not None else float("nan"),
                        "Eps": K_BASE,
                    })

                    append_sign_audit_rows(sign_audit_rows, ticker, "ENSEMBLE", ens_preds, ens_actuals, EPS_FLOOR)

                    if ens_weights:
                        for rec in ens_weights:
                            rec.update({"ticker": ticker})
                            weights_rows.append(rec)

                    sweep = k_sweep_metrics(ens_actuals, ens_signals, ens_regimes, K_BASE, [0.25, 0.5, 0.75, 1.0])
                    for rec in sweep:
                        rec.update({"ticker": ticker, "model": "ENSEMBLE"})
                        k_rows.append(rec)

                    best = max(sweep, key=lambda r: r["avg_signed_return"] if np.isfinite(r["avg_signed_return"]) else -np.inf)
                    k_best = float(best["k"])

                    base_metrics = {
                        "coverage": float(np.mean((np.array(ens_signals) > k_best) & np.array(ens_regimes, dtype=bool))),
                        "da": float(np.mean(np.array(ens_actuals)[(np.array(ens_signals) > k_best) & np.array(ens_regimes, dtype=bool)] > 0)) if np.any((np.array(ens_signals) > k_best) & np.array(ens_regimes, dtype=bool)) else float("nan"),
                        "mase": mase(ens_actuals, ens_preds, scale_train),
                    }
                    tradable = (
                        base_metrics["coverage"] >= 0.50
                        and (base_metrics["da"] is not None and base_metrics["da"] >= 0.55)
                        and (base_metrics["mase"] is not None and base_metrics["mase"] < 1.0)
                    )

                    if not tradable:
                        k_best = float("inf")

                    sim = trade_simulation(ens_actuals, ens_signals, ens_sigmas, ens_regimes, k_best, HORIZON, cost_bps=5, target_vol=0.10, min_hold=1)
                    sim.update({
                        "ticker": ticker,
                        "model": "ENSEMBLE",
                        "k_best": k_best,
                        "coverage": base_metrics["coverage"],
                        "da": base_metrics["da"],
                        "mase": base_metrics["mase"],
                        "tradable": tradable,
                    })
                    trade_rows.append(sim)
            except Exception as e:
                log_line(f"[FAIL] {ticker} ENSEMBLE -> {repr(e)}", logfile)

            if gbm_log or arima_log or naive_log or drift_log:
                with open(log_jsonl, "a", encoding="utf-8") as f:
                    for rec in drift_log + gbm_log + arima_log + naive_log:
                        f.write(json.dumps(rec) + "\n")

            log_line(
                f"[DONE] {ticker}: DRIFT_ok={len(drift_actuals)}/{len(drift_log)}, "
                f"GBM_ok={len(gbm_actuals)}/{len(gbm_log)}, "
                f"ARIMA_ok={len(arima_actuals)}/{len(arima_log)}, "
                f"NAIVE_ok={len(naive_actuals)}/{len(naive_log)}",
                logfile,
            )

            results = write_results(rows, out_csv)

    except KeyboardInterrupt:
        log_line("[INTERRUPT] Run cancelled by user. Writing partial results.", logfile)
    except Exception as e:
        log_line(f"[ERROR] Unexpected error -> {repr(e)}", logfile)
    finally:
        results = write_results(rows, out_csv)
        if sign_audit_rows:
            pd.DataFrame(sign_audit_rows).to_csv(sign_csv, index=False)
        if weights_rows:
            pd.DataFrame(weights_rows).to_csv(weights_csv, index=False)
        if k_rows:
            pd.DataFrame(k_rows).to_csv(k_csv, index=False)
        if trade_rows:
            pd.DataFrame(trade_rows).to_csv(trade_csv, index=False)
        if results is None:
            log_line("No results generated.", logfile)
        else:
            log_line(f"\nSaved results: {out_csv}", logfile)
            log_line(f"Saved split log: {log_jsonl}", logfile)
            log_line(f"Saved sign audit: {sign_csv}", logfile)
            log_line(f"Saved weights: {weights_csv}", logfile)
            log_line(f"Saved k sweep: {k_csv}", logfile)
            log_line(f"Saved trade sim: {trade_csv}", logfile)
            log_line("\nRESULTS PREVIEW:\n" + results.to_string(index=False), logfile)

if __name__ == "__main__":
    main()
