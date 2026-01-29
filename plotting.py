import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_pred_vs_actual_plot(ticker: str, model: str, preds, actuals, outdir="outputs/plots", target_label="Return"):
    os.makedirs(outdir, exist_ok=True)
    preds = np.array(preds, dtype=float)
    actuals = np.array(actuals, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title(f"{ticker} - {model} (Walk-forward terminal forecasts)")
    plt.xlabel("Split index")
    plt.ylabel(target_label)
    plt.legend()
    path = os.path.join(outdir, f"{ticker}_{model}_pred_vs_actual.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def save_error_hist_plot(ticker: str, model: str, preds, actuals, outdir="outputs/plots", target_label="Return"):
    os.makedirs(outdir, exist_ok=True)
    errs = (np.array(preds, dtype=float) - np.array(actuals, dtype=float))

    plt.figure(figsize=(10, 6))
    plt.hist(errs, bins=30)
    plt.title(f"{ticker} - {model} Error Histogram (Pred - Actual)")
    plt.xlabel(f"Error ({target_label})")
    plt.ylabel("Count")
    path = os.path.join(outdir, f"{ticker}_{model}_error_hist.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
