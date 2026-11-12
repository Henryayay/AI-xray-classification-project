import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

from cxr import calibration as cal
from cxr.metrics import all_metrics, bootstrap_ci, expected_calibration_error, wilson_ci
from cxr.stats import (delong_auc_ci, delong_test, fast_metric, holm, mcnemar_exact,
                       paired_bootstrap)


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 150)
    a = rng.normal(0, 1, 300) + 1.2 * y
    b = 0.6 * a + rng.normal(0, 1, 300) + 0.3 * y
    return y, a, b


def test_threshold_metrics_match_sklearn(data):
    y, a, _ = data
    m = all_metrics(y, a, threshold=0.3)
    yhat = (a >= 0.3).astype(int)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, yhat))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, yhat))
    assert m["f1"] == pytest.approx(f1_score(y, yhat))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, yhat))
    assert m["auroc"] == pytest.approx(roc_auc_score(y, a))


def test_fast_metric_matches(data):
    y, a, _ = data
    for metric in ("auroc", "accuracy", "balanced_accuracy", "sensitivity", "specificity"):
        assert fast_metric(metric, y, a, 0.1) == pytest.approx(all_metrics(y, a, 0.1)[metric])


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0)
    assert expected_calibration_error(y, 1 - y.astype(float)) == pytest.approx(1)


def test_wilson_known_value():
    lo, hi = wilson_ci(81, 100)
    assert lo == pytest.approx(0.7226, abs=1e-3) and hi == pytest.approx(0.8748, abs=1e-3)


def test_delong_matches_sklearn_and_bootstrap(data):
    y, a, _ = data
    auc, (lo, hi) = delong_auc_ci(y, a)
    assert auc == pytest.approx(roc_auc_score(y, a))
    blo, bhi = bootstrap_ci(y, a, "auroc", n_boot=1500, seed=0)
    assert abs(lo - blo) < 0.02 and abs(hi - bhi) < 0.02


def test_delong_identical_scores_not_significant(data):
    y, a, _ = data
    r = delong_test(y, a, a)
    assert r["diff"] == 0 and r["p"] == 1.0


def test_delong_detects_large_difference(data):
    y, a, _ = data
    noise = np.random.default_rng(3).normal(0, 1, len(y))
    assert delong_test(y, a, noise)["p"] < 1e-6


def test_mcnemar_exact():
    y = np.ones(20, int)
    pa = np.ones(20, int)
    pb = np.r_[np.zeros(10, int), np.ones(10, int)]
    r = mcnemar_exact(y, pa, pb)
    assert r["a_only_correct"] == 10 and r["b_only_correct"] == 0
    assert r["p"] == pytest.approx(2 * 0.5 ** 10)


def test_paired_bootstrap_null_and_alt(data):
    y, a, b = data
    null = paired_bootstrap(y, a[None], a[None], n_boot=300, seed=0)
    assert null["diff"] == 0 and null["p"] == 1.0
    alt = paired_bootstrap(y, np.vstack([a, a + 0.01]), b[None], n_boot=500, seed=0)
    assert alt["diff"] > 0 and alt["ci_low"] < alt["diff"] < alt["ci_high"]


def test_holm():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_calibration_shifts():
    z = np.array([1.0, 2.0, 3.0])
    assert np.allclose(cal.contextual_calibration(z, np.full(3, 2.0)), [-1, 0, 1])
    t, v = cal.adjusted_logits("val_centered", z, val_logit=np.array([0.0, 4.0]))
    assert np.allclose(t, z - 2) and np.allclose(v, [-2, 2])
    assert cal.adjusted_logits("contextual", z, logit_cf=None)[0] is None


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-true))).astype(int)
    assert cal.fit_temperature(y, 3 * true) == pytest.approx(3, rel=0.15)


def test_youden_threshold_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    z = np.array([-3, -2, -1, 4, 5, 6.0])
    t = cal.decision_threshold("youden_val", y, z)
    assert -1 < t <= 4
