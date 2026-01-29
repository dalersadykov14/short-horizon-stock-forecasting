# Short-Horizon Stock Forecasting (30-Day)

This repository contains an independent research project evaluating whether short-horizon (30 trading day) equity price movements can be forecasted using **price-only historical data**.

The work emphasizes **rigorous walk-forward evaluation**, **baseline comparison**, and **transparent failure analysis**, rather than performance optimization or trading claims.

---

## Project Overview

The project investigates short-horizon forecasting across major equities and index ETFs using:

- Statistical time-series models
- Stochastic simulation methods
- Naive baseline strategies
- A gated ensemble framework

Models are evaluated under strict **out-of-sample walk-forward validation**. The absence of reliable predictive signal is treated as a valid and important result.

---

## Models Implemented

- **Naive baselines**
  - Zero-return
  - Drift-based forecasts
- **Geometric Brownian Motion (GBM)**
- **ARIMA**
- **Weighted ensemble**
  - Constructed only from models that survive evaluation
  - Subject to confidence gating and abstention logic

---

## Methodology

- Forecast horizon: **30 trading days**
- Validation: **expanding-window walk-forward**
- Assets tested: AAPL, MSFT, NVDA, TSLA, SPY, VOO
- Metrics:
  - RMSE / MAE
  - Directional accuracy with coverage
  - Skill vs naive baselines
- Decision-level evaluation:
  - Signal gating
  - Volatility-scaled thresholds
  - Walk-forward trade simulation (with transaction costs)

All evaluation is performed strictly on unseen future data.

---

## Repository Structure

```text
.
├── backtest.py            # Walk-forward evaluation engine
├── data.py                # Data loading and preprocessing
├── metrics.py             # Error and skill metrics
├── models_arima.py        # ARIMA implementation
├── models_gbm.py          # GBM simulation model
├── models_naive.py        # Baseline models
├── plotting.py            # Diagnostic plots
├── run_experiment.py      # Main experiment runner
├── tests/
│   └── test_leakage.py    # Data leakage safeguards
├── outputs/
│   ├── logs/              # Per-split logs
│   ├── plots/             # Generated figures
│   └── tables/            # CSV result tables
└── requirements.txt
