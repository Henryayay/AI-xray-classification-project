"""Post-hoc score adjustment fitted on validation predictions only.

All functions operate on positive-class logits. For a two-class verbaliser the
logit is ``log p(positive label) - log p(negative label)``.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import youden_threshold


def contextual_calibration(logit, logit_content_free):
    """Zhao et al. (2021): remove the bias the prompt induces on a content-free input.

    For two classes, dividing probabilities by the content-free probabilities and
    renormalising is exactly a shift of the logit by the content-free logit.
    """
    return np.asarray(logit, float) - np.asarray(logit_content_free, float)


def fit_bias_shift(val_logit):
    """Validation-estimated centering in the spirit of batch calibration (Zhou et al., 2024).

    Estimating the mean on validation rather than the test batch keeps the test
    split untouched; with balanced classes this removes systematic label bias.
    """
    return float(np.mean(val_logit))


def fit_temperature(val_y, val_logit):
    """Single-parameter temperature scaling minimising validation negative log-likelihood."""
    y = np.asarray(val_y, float)
    z = np.asarray(val_logit, float)

    def nll(log_t):
        t = np.exp(log_t)
        s = z / t
        return float(np.mean(np.logaddexp(0, s) - y * s))

    res = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
    return float(np.exp(res.x))


def fit_platt(val_y, val_logit):
    """Two-parameter logistic recalibration ``a * logit + b`` fitted on validation."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.asarray(val_logit, float)[:, None], np.asarray(val_y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


SCORE_VARIANTS = ("raw", "contextual", "val_centered")


def adjusted_logits(variant, logit, logit_cf=None, val_logit=None, val_logit_cf=None):
    """Return (adjusted test logit, adjusted val logit) for a calibration variant."""
    logit = np.asarray(logit, float)
    val_logit = None if val_logit is None else np.asarray(val_logit, float)
    if variant == "raw":
        return logit, val_logit
    if variant == "contextual":
        if logit_cf is None or np.all(np.isnan(logit_cf)):
            return None, None
        v = None if val_logit is None else contextual_calibration(val_logit, val_logit_cf)
        return contextual_calibration(logit, logit_cf), v
    if variant == "val_centered":
        if val_logit is None:
            return None, None
        b = fit_bias_shift(val_logit)
        return logit - b, val_logit - b
    raise ValueError(variant)


def decision_threshold(rule, val_y=None, val_logit=None):
    """Logit threshold for hard predictions. ``default`` is the argmax rule (0)."""
    if rule == "default":
        return 0.0
    if rule == "youden_val":
        return youden_threshold(val_y, val_logit)
    raise ValueError(rule)
