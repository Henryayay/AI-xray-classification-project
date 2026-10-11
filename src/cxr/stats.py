"""Paired hypothesis tests and multiplicity correction."""

import numpy as np
from scipy import stats as sps

from .metrics import all_metrics, stratified_bootstrap_indices


def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(y, scores):
    """Structural components of DeLong's AUROC variance (Sun & Xu, 2014)."""
    y = np.asarray(y).astype(int)
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def delong_auc_ci(y, score, alpha=0.05):
    aucs, cov = _delong_components(y, np.asarray(score, dtype=float)[None, :])
    se = float(np.sqrt(cov[0, 0]))
    z = sps.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), (float(aucs[0] - z * se), float(aucs[0] + z * se))


def delong_test(y, score_a, score_b):
    """Two-sided DeLong test for two correlated AUROCs on the same cases."""
    scores = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    aucs, cov = _delong_components(y, scores)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": 0.0,
                "p": 1.0}
    z = diff / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z),
            "p": float(2 * sps.norm.sf(abs(z)))}


def mcnemar_exact(y, pred_a, pred_b):
    """Exact (binomial) McNemar test on paired correctness."""
    y = np.asarray(y)
    ca = np.asarray(pred_a) == y
    cb = np.asarray(pred_b) == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = 1.0 if b + c == 0 else float(sps.binomtest(b, b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p": p}


def paired_bootstrap(y, logits_a, logits_b, metric="auroc", thresholds_a=None,
                     thresholds_b=None, n_boot=10000, seed=0, alpha=0.05):
    """Paired, class-stratified bootstrap of the difference in seed-averaged metric.

    ``logits_a`` and ``logits_b`` are arrays of shape (n_seeds, n_cases); the
    two methods may have different numbers of seeds. Each replicate resamples
    cases once and evaluates every seed of both methods on the same resample,
    so the interval reflects case-sampling uncertainty of the seed-mean
    difference. The two-sided p-value is ``2 * min(P(d <= 0), P(d >= 0))``.
    """
    y = np.asarray(y)
    la = np.atleast_2d(np.asarray(logits_a, float))
    lb = np.atleast_2d(np.asarray(logits_b, float))
    ta = np.zeros(len(la)) if thresholds_a is None else np.asarray(thresholds_a, float)
    tb = np.zeros(len(lb)) if thresholds_b is None else np.asarray(thresholds_b, float)

    def seed_mean(L, T, idx):
        return np.mean([all_metrics(y[idx], L[s, idx], T[s])[metric] for s in range(len(L))])

    full = np.arange(len(y))
    observed = seed_mean(la, ta, full) - seed_mean(lb, tb, full)
    diffs = np.array([
        seed_mean(la, ta, idx) - seed_mean(lb, tb, idx)
        for idx in stratified_bootstrap_indices(y, n_boot, seed)
    ])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return {"diff": float(observed), "ci_low": float(lo), "ci_high": float(hi), "p": float(p),
            "n_boot": int(n_boot)}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
