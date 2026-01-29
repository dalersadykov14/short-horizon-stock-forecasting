import numpy as np

def rmse(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def nrmse(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    denom = float(np.mean(y_true))
    return rmse(y_true, y_pred) / denom if denom != 0 else float("nan")

def nmae(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    denom = float(np.mean(y_true))
    return mae(y_true, y_pred) / denom if denom != 0 else float("nan")

def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def mase(y_true, y_pred, y_train) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_train = np.array(y_train, dtype=float)
    if y_train.size < 2:
        return float("nan")
    scale = float(np.mean(np.abs(np.diff(y_train))))
    return mae(y_true, y_pred) / scale if scale != 0 else float("nan")

def rmsse(y_true, y_pred, y_train) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_train = np.array(y_train, dtype=float)
    if y_train.size < 2:
        return float("nan")
    scale = float(np.mean(np.square(np.diff(y_train))))
    return rmse(y_true, y_pred) / np.sqrt(scale) if scale != 0 else float("nan")

def _sign_labels(y, eps: float) -> np.ndarray:
    y = np.array(y, dtype=float)
    labels = np.zeros_like(y, dtype=int)
    labels[y > eps] = 1
    labels[y < -eps] = -1
    return labels

def sign_audit(y_pred, eps: float) -> dict:
    y_pred = np.array(y_pred, dtype=float)
    if y_pred.size == 0:
        return {
            "pos_rate": float("nan"),
            "neg_rate": float("nan"),
            "neutral_rate": float("nan"),
            "mean_pred": float("nan"),
            "std_pred": float("nan"),
            "coverage": float("nan"),
        }

    labels = _sign_labels(y_pred, eps)
    pos_rate = float(np.mean(labels == 1))
    neg_rate = float(np.mean(labels == -1))
    neutral_rate = float(np.mean(labels == 0))
    coverage = float(np.mean(labels != 0))

    return {
        "pos_rate": pos_rate,
        "neg_rate": neg_rate,
        "neutral_rate": neutral_rate,
        "mean_pred": float(np.mean(y_pred)),
        "std_pred": float(np.std(y_pred, ddof=1)) if y_pred.size > 1 else 0.0,
        "coverage": coverage,
    }

def directional_accuracy_neutral(y_true, y_pred, eps: float) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")

    pred_labels = _sign_labels(y_pred, eps)
    true_labels = _sign_labels(y_true, eps)

    mask = (pred_labels != 0) & (true_labels != 0)
    if not np.any(mask):
        return float("nan")

    return float(np.mean(pred_labels[mask] == true_labels[mask]))

def balanced_accuracy_posneg(y_true, y_pred, eps: float) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")

    pred_labels = _sign_labels(y_pred, eps)
    true_labels = _sign_labels(y_true, eps)

    mask = true_labels != 0
    if not np.any(mask):
        return float("nan")

    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]

    pos_mask = true_labels == 1
    neg_mask = true_labels == -1

    tpr = float(np.mean(pred_labels[pos_mask] == 1)) if np.any(pos_mask) else float("nan")
    tnr = float(np.mean(pred_labels[neg_mask] == -1)) if np.any(neg_mask) else float("nan")

    if np.isnan(tpr) or np.isnan(tnr):
        return float("nan")
    return 0.5 * (tpr + tnr)
