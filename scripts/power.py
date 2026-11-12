#!/usr/bin/env python
"""Precision and minimum detectable effects for the fixed test set (n = 456, 228 per class).

The sample size is fixed by the published split, so instead of choosing n we
report what it can resolve:

* Wilson 95% CI half-width of accuracy / sensitivity / specificity;
* Hanley-McNeil standard error and CI half-width of AUROC;
* minimum detectable AUROC difference (80% power, two-sided) for two
  correlated AUROCs on the same cases, for a range of inter-model correlation;
* minimum detectable paired accuracy difference for McNemar's test (Connor,
  1987) as a function of the discordant-pair rate;

each at alpha = 0.05 and at the Holm worst case alpha / m for the m primary tests.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.metrics import wilson_ci  # noqa: E402

N_POS = N_NEG = 228


def hanley_mcneil_se(auc, n1=N_POS, n0=N_NEG):
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
    return float(np.sqrt(var))


def auc_mde(auc, r, alpha, power=0.8):
    se = hanley_mcneil_se(auc)
    se_diff = np.sqrt(2 * se**2 * (1 - r))
    return float((sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)) * se_diff)


def mcnemar_mde(n, psi, alpha, power=0.8):
    """Smallest |p10 - p01| detectable with McNemar's test (Connor 1987 approximation)."""
    za, zb = sps.norm.ppf(1 - alpha / 2), sps.norm.ppf(power)

    def f(d):
        return (za * np.sqrt(psi) + zb * np.sqrt(psi - d**2)) ** 2 / d**2 - n

    hi = np.sqrt(psi) - 1e-9
    if f(hi) > 0:
        return float("nan")
    return float(brentq(f, 1e-6, hi))


def main():
    plan = yaml.safe_load((ROOT / "configs" / "analysis_plan.yaml").read_text())
    m = len(plan["primary"]) * 2
    alphas = {"alpha_0.05": 0.05, f"holm_worst_case_0.05/{m}": 0.05 / m}
    n = N_POS + N_NEG
    rows_acc = [{"metric_value": p, "n": n, "ci_half_width": (lambda c: (c[1] - c[0]) / 2)(
        wilson_ci(int(round(p * n)), n)), "per_class_ci_half_width": (lambda c: (c[1] - c[0]) / 2)(
        wilson_ci(int(round(p * N_POS)), N_POS))} for p in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)]
    rows_auc = []
    for auc in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        se = hanley_mcneil_se(auc)
        row = {"auroc": auc, "se": se, "ci_half_width": 1.96 * se}
        for name, a in alphas.items():
            for r in (0.3, 0.5, 0.7):
                row[f"mde_r{r}_{name}"] = auc_mde(auc, r, a)
        rows_auc.append(row)
    rows_mc = []
    for psi in (0.05, 0.1, 0.15, 0.2, 0.3):
        row = {"discordant_rate": psi}
        for name, a in alphas.items():
            row[f"mde_{name}"] = mcnemar_mde(n, psi, a)
        rows_mc.append(row)
    out = ROOT / "results" / "power"
    out.mkdir(parents=True, exist_ok=True)
    acc, auc, mc = pd.DataFrame(rows_acc), pd.DataFrame(rows_auc), pd.DataFrame(rows_mc)
    acc.to_csv(out / "accuracy_precision.csv", index=False)
    auc.to_csv(out / "auroc_precision_mde.csv", index=False)
    mc.to_csv(out / "mcnemar_mde.csv", index=False)
    (out / "power.json").write_text(json.dumps({
        "n_test": n, "n_pos": N_POS, "n_neg": N_NEG, "n_primary_tests": m, "power": 0.8,
        "accuracy": rows_acc, "auroc": rows_auc, "mcnemar": rows_mc}, indent=2))
    print(acc.round(4).to_string(), "\n", auc.round(4).to_string(), "\n", mc.round(4).to_string())


if __name__ == "__main__":
    main()
