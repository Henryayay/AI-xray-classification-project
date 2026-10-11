#!/usr/bin/env python
"""Turn logged predictions into metrics, pre-registered comparisons and summaries.

Reads every ``results/<task>/<experiment>/<run>/predictions.parquet`` and writes
``results/summary/``:

  run_metrics.csv           one row per run x score variant x decision rule (val and test)
  condition_summary.csv     mean / SD / min / max over seeds or draws per condition
  selections.json           validation-selected condition for every family
  comparisons.csv           primary (Holm-adjusted) and secondary comparisons on test
  robustness.csv            test AUROC under each perturbation and drop vs. unperturbed
  leakage_sensitivity.csv   test AUROC with/without test images that have a highly similar
                            training image (from scripts/audit_dataset.py)
  generation_vs_likelihood.csv   original string-parsing protocol vs. likelihood scoring
  summary.json              everything above in one file for table/figure generation

Only runs produced with ``--final`` and without ``--limit`` enter test-set analyses.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr import calibration as cal  # noqa: E402
from cxr.io import load_runs  # noqa: E402
from cxr.metrics import all_metrics, sigmoid, expected_calibration_error, wilson_ci  # noqa: E402
from cxr.stats import delong_auc_ci, delong_test, holm, mcnemar_exact, paired_bootstrap  # noqa: E402

RULES = ("default", "youden_val")
COND_KEYS = ["task", "experiment", "train_fraction", "shots"]


def split_frame(preds, split, perturbation="none"):
    f = preds[(preds.split == split) & (preds.perturbation == perturbation)]
    return f.sort_values("index").reset_index(drop=True)


def _cf(f):
    return f["logit_cf"].to_numpy(float) if "logit_cf" in f else None


def variants_for(run):
    preds = run["predictions"]
    has_cf = "logit_cf" in preds and preds["logit_cf"].notna().any()
    is_vlm = str(run["meta"].get("kind", "")).startswith("vlm") or run["experiment"].startswith(
        "ensemble")
    out = ["raw"]
    if is_vlm:
        out += (["contextual"] if has_cf else []) + ["val_centered"]
    return out


def scored(run, variant, split="test", perturbation="none"):
    """(y, adjusted logit, adjusted val logit, val y) for one run."""
    preds = run["predictions"]
    val = split_frame(preds, "validation")
    tgt = split_frame(preds, split, perturbation)
    if tgt.empty:
        return None
    lt, lv = cal.adjusted_logits(variant, tgt.logit.to_numpy(float), _cf(tgt),
                                 val.logit.to_numpy(float), _cf(val))
    if lt is None:
        return None
    return tgt.y_true.to_numpy(int), lt, lv, val.y_true.to_numpy(int), tgt["index"].to_numpy()


def is_final(run):
    m = run["meta"]
    return bool(m.get("final")) and not m.get("limit")


def run_key(run):
    m = run["meta"]
    shots = m.get("shots")
    return {"task": run["task"], "experiment": run["experiment"], "run": run["run"],
            "seed": m.get("seed"), "train_fraction": float(m.get("train_fraction", 1.0)),
            "shots": -1 if shots is None or (isinstance(shots, float) and np.isnan(shots))
            else int(shots)}


def per_run_metrics(runs):
    rows = []
    for run in runs:
        for variant in variants_for(run):
            s_val = scored(run, variant, "validation")
            if s_val is None:
                continue
            yv, lv = s_val[0], s_val[1]
            T = cal.fit_temperature(yv, lv)
            for rule in RULES:
                thr = cal.decision_threshold(rule, yv, lv)
                for split in ("validation", "test"):
                    if split == "test" and not is_final(run):
                        continue
                    s = scored(run, variant, split)
                    if s is None:
                        continue
                    y, lt = s[0], s[1]
                    m = all_metrics(y, lt, thr)
                    m["ece_temp_scaled"] = expected_calibration_error(y, sigmoid(lt / T))
                    m["temperature"] = T
                    _, (lo, hi) = delong_auc_ci(y, lt)
                    m["auroc_ci_low"], m["auroc_ci_high"] = lo, hi
                    k = int(round(m["accuracy"] * len(y)))
                    m["accuracy_ci_low"], m["accuracy_ci_high"] = wilson_ci(k, len(y))
                    rows.append({**run_key(run), "variant": variant, "rule": rule,
                                 "threshold": thr, "split": split, **m})
    return pd.DataFrame(rows)


def condition_summary(rm):
    if rm.empty:
        return rm
    metrics = ["auroc", "auprc", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
               "f1", "mcc", "brier", "ece", "ece_temp_scaled", "positive_rate"]
    g = rm.groupby(COND_KEYS + ["variant", "rule", "split"])
    out = g[metrics].agg(["mean", "std", "min", "max"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()


# ------------------------------------------------------------------ selection

def select_condition(cs, task, spec, plan):
    """Validation-best condition for a family (mean val AUROC, raw scores, full training data)."""
    exps = [spec["experiment"]] if "experiment" in spec else plan["families"][spec["family"]]
    c = cs[(cs.task == task) & cs.experiment.isin(exps) & (cs.split == "validation")
           & (cs.variant == "raw") & (cs.rule == "default") & (cs.train_fraction == 1.0)]
    if c.empty:
        return None
    best = c.sort_values(["auroc_mean", "experiment", "shots"], ascending=[False, True, True])
    r = best.iloc[0]
    return {"experiment": r.experiment, "train_fraction": float(r.train_fraction),
            "shots": int(r.shots), "val_auroc_mean": float(r.auroc_mean),
            "n_candidates": int(len(c))}


def condition_runs(runs, task, cond):
    sel = [r for r in runs if r["task"] == task and r["experiment"] == cond["experiment"]
           and run_key(r)["train_fraction"] == cond["train_fraction"]
           and run_key(r)["shots"] == cond["shots"] and is_final(r)]
    return sorted(sel, key=lambda r: r["run"])


def stacked(runs_, variant, rule, perturbation="none"):
    Ls, Ts, y, idx0 = [], [], None, None
    for r in runs_:
        s = scored(r, variant, "test", perturbation)
        if s is None:
            return None
        yy, lt, lv, yv, idx = s
        if idx0 is not None and not np.array_equal(idx, idx0):
            raise ValueError("test indices differ between runs")
        y, idx0 = yy, idx
        Ls.append(lt)
        Ts.append(cal.decision_threshold(rule, yv, lv))
    return (np.array(Ls), np.array(Ts), y) if Ls else None


# ------------------------------------------------------------------- ensemble

def build_ensembles(runs, cs, plan, tasks):
    """Validation-weighted average of z-scored logits: best VLM + best supervised model."""
    new = []
    for task in tasks:
        a = select_condition(cs, task, {"family": "vlm_any"}, plan)
        b = select_condition(cs, task, {"family": "supervised"}, plan)
        if not a or not b:
            continue
        ra, rb = condition_runs(runs, task, a), condition_runs(runs, task, b)
        for k, (x, z) in enumerate(zip(ra, rb)):
            px, pz = x["predictions"], z["predictions"]
            vx, vz = split_frame(px, "validation"), split_frame(pz, "validation")
            tx, tz = split_frame(px, "test"), split_frame(pz, "test")
            assert np.array_equal(vx["index"], vz["index"]) and np.array_equal(tx["index"],
                                                                                tz["index"])

            def zs(v, t):
                mu, sd = v.logit.mean(), v.logit.std() + 1e-9
                return (v.logit.to_numpy() - mu) / sd, (t.logit.to_numpy() - mu) / sd

            zvx, ztx = zs(vx, tx)
            zvz, ztz = zs(vz, tz)
            yv = vx.y_true.to_numpy()
            grid = np.round(np.linspace(0, 1, 11), 2)
            w = max(grid, key=lambda w_: (all_metrics(yv, w_ * zvx + (1 - w_) * zvz)["auroc"],
                                          -abs(w_ - 0.5)))
            preds = pd.concat([
                pd.DataFrame({"split": "validation", "index": vx["index"], "y_true": yv,
                              "logit": w * zvx + (1 - w) * zvz, "perturbation": "none"}),
                pd.DataFrame({"split": "test", "index": tx["index"], "y_true": tx.y_true,
                              "logit": w * ztx + (1 - w) * ztz, "perturbation": "none"}),
            ])
            new.append({"task": task, "experiment": "ensemble_vlm_supervised",
                        "run": f"pair{k}", "predictions": preds,
                        "meta": {"kind": "ensemble", "final": True, "seed": k, "shots": None,
                                 "train_fraction": 1.0, "weight_vlm": float(w),
                                 "components": [x["experiment"] + "/" + x["run"],
                                                z["experiment"] + "/" + z["run"]]}})
    return new


# ---------------------------------------------------------------- comparisons

def compare(runs, cs, plan, task, comp, n_boot, seed):
    a = select_condition(cs, task, comp["a"], plan)
    b_spec = comp["b"]
    b = dict(a) if b_spec.get("same_condition_as") == "a" and a else select_condition(
        cs, task, b_spec, plan)
    base = {"task": task, "id": comp["id"], "metric": comp["metric"]}
    if not a or not b:
        return {**base, "status": "missing conditions"}
    va, vb = comp["a"].get("variant", "raw"), b_spec.get("variant", "raw")
    rule_a, rule_b = comp["a"].get("rule", "default"), b_spec.get("rule", "default")
    sa = stacked(condition_runs(runs, task, a), va, rule_a)
    sb = stacked(condition_runs(runs, task, b), vb, rule_b)
    if sa is None or sb is None:
        return {**base, "status": "missing test predictions"}
    La, Ta, y = sa
    Lb, Tb, yb = sb
    assert np.array_equal(y, yb)
    pb = paired_bootstrap(y, La, Lb, comp["metric"], Ta, Tb, n_boot=n_boot, seed=seed)
    dl = delong_test(y, La.mean(0), Lb.mean(0))
    mc = mcnemar_exact(y, (La[0] >= Ta[0]).astype(int), (Lb[0] >= Tb[0]).astype(int))
    return {**base, "status": "ok",
            "a_condition": f"{a['experiment']}|shots={a['shots']}|{va}|{rule_a}",
            "b_condition": f"{b['experiment']}|shots={b['shots']}|{vb}|{rule_b}",
            "a_runs": len(La), "b_runs": len(Lb), **{f"boot_{k}": v for k, v in pb.items()},
            "delong_seedmean_diff": dl["diff"], "delong_seedmean_p": dl["p"],
            "mcnemar_run0_p": mc["p"], "mcnemar_run0_a_only": mc["a_only_correct"],
            "mcnemar_run0_b_only": mc["b_only_correct"]}


def robustness(runs):
    rows = []
    for run in runs:
        if not is_final(run):
            continue
        perts = sorted(set(run["predictions"].perturbation) - {"none"})
        base = scored(run, "raw", "test")
        if base is None or not perts:
            continue
        a0 = all_metrics(base[0], base[1])["auroc"]
        for p in perts:
            s = scored(run, "raw", "test", p)
            a = all_metrics(s[0], s[1])["auroc"]
            rows.append({**run_key(run), "perturbation": p, "auroc": a, "auroc_clean": a0,
                         "auroc_drop": a0 - a})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    agg = df.groupby(COND_KEYS + ["perturbation"]).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        drop_mean=("auroc_drop", "mean"), drop_std=("auroc_drop", "std"),
        n_runs=("auroc", "size")).reset_index()
    return df, agg


def leakage_exclusions(results_root):
    """Test indices with a highly-similar-or-closer training image, per task (from the audit)."""
    out = {}
    for f in Path(results_root, "audit").glob("duplicates_task*.csv"):
        d = pd.read_csv(f)
        task = f.stem.replace("duplicates_", "")
        idx = set(d.loc[(d.split_a == "test") & (d.split_b == "train"), "index_a"]) | set(
            d.loc[(d.split_b == "test") & (d.split_a == "train"), "index_b"])
        out[task] = sorted(int(i) for i in idx)
    return out


def leakage_sensitivity(runs, exclusions):
    rows = []
    for run in runs:
        ex = exclusions.get(run["task"])
        if not ex or not is_final(run):
            continue
        s = scored(run, "raw", "test")
        if s is None:
            continue
        y, lt, _, _, idx = s
        keep = ~np.isin(idx, ex)
        rows.append({**run_key(run), "n_excluded": int((~keep).sum()),
                     "auroc_all": all_metrics(y, lt)["auroc"],
                     "auroc_excluding_similar": all_metrics(y[keep], lt[keep])["auroc"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["auroc_change"] = df.auroc_excluding_similar - df.auroc_all
    return df.groupby(COND_KEYS).agg(
        n_excluded=("n_excluded", "first"), auroc_all=("auroc_all", "mean"),
        auroc_excluding_similar=("auroc_excluding_similar", "mean"),
        change_mean=("auroc_change", "mean"), change_max_abs=("auroc_change",
                                                              lambda v: v.abs().max()),
        n_runs=("auroc_all", "size")).reset_index()


def generation_vs_likelihood(runs):
    rows = []
    for run in runs:
        p = run["predictions"]
        if "parsed_label" not in p:
            continue
        for split in ("validation", "test"):
            if split == "test" and not is_final(run):
                continue
            f = split_frame(p, split)
            f = f[f.parsed_label != -100]
            if f.empty:
                continue
            y = f.y_true.to_numpy()
            parsed = f.parsed_label.to_numpy()
            lik = (f.logit.to_numpy() >= 0).astype(int)
            rows.append({**run_key(run), "split": split, "n": len(f),
                         "parse_failure_rate": float(np.mean(parsed == -1)),
                         "accuracy_generation": float(np.mean(parsed == y)),
                         "accuracy_likelihood": float(np.mean(lik == y)),
                         "agreement": float(np.mean(parsed == lik))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--plan", default=str(ROOT / "configs" / "analysis_plan.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text())
    n_boot = args.n_boot or plan["n_boot"]
    out = Path(args.out or Path(args.results) / "summary")
    out.mkdir(parents=True, exist_ok=True)

    runs = [r for r in load_runs(args.results) if r["task"] not in ("_smoke", "audit")]
    tasks = sorted({r["task"] for r in runs})
    print(f"{len(runs)} runs across tasks {tasks}")
    rm = per_run_metrics(runs)
    cs = condition_summary(rm)
    ens = build_ensembles(runs, cs, plan, tasks)
    if ens:
        rm = pd.concat([rm, per_run_metrics(ens)], ignore_index=True)
        cs = condition_summary(rm)
        runs = runs + ens
    rm.to_csv(out / "run_metrics.csv", index=False)
    cs.to_csv(out / "condition_summary.csv", index=False)

    families = {**plan["families"], "ensemble": ["ensemble_vlm_supervised"]}
    selections = {t: {f: select_condition(cs, t, {"family": f}, {"families": families})
                      for f in families} for t in tasks} if not cs.empty else {}
    (out / "selections.json").write_text(json.dumps(selections, indent=2))

    comps = []
    for tier in ("primary", "secondary"):
        for comp in plan.get(tier, []):
            for t in tasks:
                if cs.empty:
                    continue
                r = compare(runs, cs, plan, t, comp, n_boot, plan["bootstrap_seed"])
                r["tier"] = tier
                comps.append(r)
    cdf = pd.DataFrame(comps)
    if not cdf.empty and "boot_p" in cdf:
        cdf["p_holm"] = np.nan
        m = (cdf.tier == "primary") & (cdf.status == "ok")
        if m.any():
            cdf.loc[m, "p_holm"] = holm(cdf.loc[m, "boot_p"].to_numpy())
    cdf.to_csv(out / "comparisons.csv", index=False)

    rob, rob_agg = robustness(runs)
    rob.to_csv(out / "robustness_runs.csv", index=False)
    rob_agg.to_csv(out / "robustness.csv", index=False)
    exclusions = leakage_exclusions(args.results)
    leakage_sensitivity(runs, exclusions).to_csv(out / "leakage_sensitivity.csv", index=False)
    gvl = generation_vs_likelihood(runs)
    gvl.to_csv(out / "generation_vs_likelihood.csv", index=False)
    ens_meta = [{"task": e["task"], "run": e["run"], **{k: e["meta"][k] for k in
                                                        ("weight_vlm", "components")}}
                for e in ens]
    (out / "summary.json").write_text(json.dumps({
        "n_runs": len(runs), "tasks": tasks, "selections": selections, "ensembles": ens_meta,
        "comparisons": cdf.to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
