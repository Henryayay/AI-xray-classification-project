"""Classification metrics with bootstrap and Wilson confidence intervals.

Scores are logits for the positive class (index 1). Probabilities are
``sigmoid(logit)``; hard predictions use ``logit >= threshold``.
"""

import numpy as np
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def confusion(y, yhat):
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    tp = int(np.sum((y == 1) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    return tp, tn, fp, fn


def _safe_div(a, b):
    return float(a) / b if b else float("nan")


def expected_calibration_error(y, p, n_bins=10):
    """Equal-width-bin ECE of the positive-class probability."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def threshold_metrics(y, yhat):
    tp, tn, fp, fn = confusion(y, yhat)
    n = tp + tn + fp + fn
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "accuracy": _safe_div(tp + tn, n),
        "balanced_accuracy": np.nanmean([sens, spec]),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "mcc": float(mcc),
        "positive_rate": _safe_div(tp + fp, n),
    }


def all_metrics(y, logit, threshold=0.0):
    y = np.asarray(y).astype(int)
    logit = np.asarray(logit, dtype=float)
    p = sigmoid(logit)
    out = threshold_metrics(y, (logit >= threshold).astype(int))
    if len(np.unique(y)) == 2 and np.ptp(logit) > 0:
        out["auroc"] = float(roc_auc_score(y, logit))
        out["auprc"] = float(average_precision_score(y, logit))
    else:
        out["auroc"] = 0.5 if len(np.unique(y)) == 2 else float("nan")
        out["auprc"] = float(np.mean(y)) if len(y) else float("nan")
    out["brier"] = float(np.mean((p - y) ** 2))
    out["ece"] = expected_calibration_error(y, p)
    out["n"] = int(len(y))
    return out


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def stratified_bootstrap_indices(y, n_boot, seed):
    """Resample positives and negatives separately so every replicate keeps both classes."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, logit, metric="auroc", threshold=0.0, n_boot=2000, alpha=0.05, seed=0):
    y = np.asarray(y)
    logit = np.asarray(logit, dtype=float)
    vals = np.array([
        all_metrics(y[i], logit[i], threshold)[metric]
        for i in stratified_bootstrap_indices(y, n_boot, seed)
    ])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def youden_threshold(y, logit):
    """Logit threshold maximising sensitivity + specificity - 1 (fit on validation only)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, logit)
    j = tpr - fpr
    best = int(np.argmax(j))
    t = thr[best]
    return float(t) if np.isfinite(t) else float(np.max(logit) + 1)
