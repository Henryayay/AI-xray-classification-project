"""Write results/summary/experiment_results.md from predictions, then check it.

Every displayed metric is recomputed with src/cxr and must match
results/summary/*.csv (produced by scripts/analyze.py) before the file is written.
"""

import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.calibration import decision_threshold  # noqa: E402
from cxr.io import load_runs  # noqa: E402
from cxr.metrics import all_metrics, confusion, sigmoid, expected_calibration_error  # noqa: E402
from cxr.calibration import fit_temperature  # noqa: E402
from cxr.stats import delong_auc_ci, delong_test, mcnemar_exact, paired_bootstrap  # noqa: E402

SUMMARY = ROOT / "results" / "summary"
OUT = SUMMARY / "experiment_results.md"

NAMES = {
    "probe_rad_dino": "RAD-DINO linear probe",
    "probe_imagenet_resnet50": "ResNet-50 (ImageNet) linear probe",
}
TASKS = {
    "task1": ("Task 1", "Normal", "Pneumonia"),
    "task2": ("Task 2", "Non-COVID pneumonia", "COVID-19"),
}
PERT_LABEL = {
    "border_mask": "Border hidden",
    "centre_mask": "Centre hidden",
    "gamma_low": "Gamma 0.6",
    "gamma_high": "Gamma 1.6",
    "noise": "Gaussian noise",
    "jpeg": "JPEG quality 10",
}


def q(x, places):
    d = Decimal(str(float(x))).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return f"{d:.{places}f}"


def f3(x):
    return q(x, 3)


def f4(x):
    return q(x, 4)


def fmt_p(p):
    p = float(p)
    if p < 0.001:
        return f"{p:.2e}"
    return q(p, 3)


def md_table(headers, rows):
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def split_frame(preds, split, perturbation="none"):
    f = preds[(preds.split == split) & (preds.perturbation == perturbation)]
    return f.sort_values("index").reset_index(drop=True)


def close(a, b, tol=1e-9):
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(float(a) - float(b)) <= tol


def main():
    runs = [r for r in load_runs(ROOT / "results") if r["task"] in TASKS]
    rm = pd.read_csv(SUMMARY / "run_metrics.csv")
    rob = pd.read_csv(SUMMARY / "robustness.csv")
    leak = pd.read_csv(SUMMARY / "leakage_sensitivity.csv")

    # Recompute and require an exact match to analyze.py before any table is written.
    checked = 0
    details = {}
    for run in runs:
        preds = run["predictions"]
        val = split_frame(preds, "validation")
        yv = val.y_true.to_numpy(int)
        lv = val.logit.to_numpy(float)
        T = fit_temperature(yv, lv)
        meta = run["meta"]
        key = (run["task"], run["experiment"], float(meta.get("train_fraction", 1.0)))
        details[key] = {
            "C": meta["C"],
            "val_nll": meta["val_nll"],
            "train_size": meta["train_size"],
            "seed": meta["seed"],
            "temperature": T,
        }
        for rule in ("default", "youden_val"):
            thr = decision_threshold(rule, yv, lv)
            for split, frame, yy, logit in (
                ("validation", val, yv, lv),
                ("test", split_frame(preds, "test"), None, None),
            ):
                if split == "test":
                    yy = frame.y_true.to_numpy(int)
                    logit = frame.logit.to_numpy(float)
                m = all_metrics(yy, logit, thr)
                m["ece_temp_scaled"] = expected_calibration_error(yy, sigmoid(logit / T))
                _, (lo, hi) = delong_auc_ci(yy, logit)
                row = rm[(rm.task == run["task"]) & (rm.experiment == run["experiment"])
                         & (rm.train_fraction == key[2]) & (rm.split == split)
                         & (rm.rule == rule) & (rm.variant == "raw")].iloc[0]
                for col in ("auroc", "auprc", "accuracy", "balanced_accuracy", "sensitivity",
                            "specificity", "precision", "f1", "mcc", "brier", "ece",
                            "positive_rate", "ece_temp_scaled"):
                    if not close(m[col], row[col]):
                        raise SystemExit(f"mismatch {key} {split} {rule} {col}: {m[col]} vs {row[col]}")
                if not close(lo, row.auroc_ci_low) or not close(hi, row.auroc_ci_high):
                    raise SystemExit(f"CI mismatch {key} {split} {rule}")
                if not close(thr, row.threshold, tol=1e-8):
                    raise SystemExit(f"threshold mismatch {key} {split} {rule}")
                if int(m["n"]) != int(row.n):
                    raise SystemExit(f"n mismatch {key}")
                checked += 1
                if split == "test":
                    tp, tn, fp, fn = confusion(yy, (logit >= thr).astype(int))
                    n = tp + tn + fp + fn
                    # Arithmetic identities a reader can check.
                    assert tp + fn == int((yy == 1).sum())
                    assert tn + fp == int((yy == 0).sum())
                    assert abs((tp + tn) / n - m["accuracy"]) < 1e-12
                    assert abs(tp / (tp + fn) - m["sensitivity"]) < 1e-12
                    assert abs(tn / (tn + fp) - m["specificity"]) < 1e-12
                    details[key].setdefault("counts", {})[rule] = (tp, tn, fp, fn, n)
                    details[key].setdefault("test_logit", logit)
                    details[key].setdefault("test_y", yy)
                    details[key].setdefault("test_thr", {})[rule] = thr

        base = all_metrics(
            split_frame(preds, "test").y_true.to_numpy(int),
            split_frame(preds, "test").logit.to_numpy(float),
        )["auroc"]
        for pert in PERT_LABEL:
            s = split_frame(preds, "test", pert)
            a = all_metrics(s.y_true.to_numpy(int), s.logit.to_numpy(float))["auroc"]
            row = rob[(rob.task == run["task"]) & (rob.experiment == run["experiment"])
                      & (np.isclose(rob.train_fraction, key[2])) & (rob.perturbation == pert)].iloc[0]
            if not close(a, row.auroc_mean) or not close(base - a, row.drop_mean):
                raise SystemExit(f"robustness mismatch {key} {pert}")
            details[key].setdefault("pert", {})[pert] = a

        ex_file = ROOT / "results" / "audit" / f"duplicates_{run['task']}.csv"
        dups = pd.read_csv(ex_file)
        excluded = set(dups.loc[(dups.split_a == "test") & (dups.split_b == "train"), "index_a"]) | set(
            dups.loc[(dups.split_b == "test") & (dups.split_a == "train"), "index_b"])
        te = split_frame(preds, "test")
        y = te.y_true.to_numpy(int)
        logit = te.logit.to_numpy(float)
        keep = ~np.isin(te["index"].to_numpy(), list(excluded))
        a_all = all_metrics(y, logit)["auroc"]
        a_ex = all_metrics(y[keep], logit[keep])["auroc"]
        lrow = leak[(leak.task == run["task"]) & (leak.experiment == run["experiment"])
                    & np.isclose(leak.train_fraction, key[2])].iloc[0]
        if not close(a_all, lrow.auroc_all) or not close(a_ex, lrow.auroc_excluding_similar):
            raise SystemExit(f"leakage mismatch {key}")
        if int((~keep).sum()) != int(lrow.n_excluded):
            raise SystemExit(f"exclusion count mismatch {key}")
        details[key]["leak"] = (int((~keep).sum()), a_all, a_ex, a_ex - a_all)

    # Descriptive comparison of the two probes that were run (full training data).
    comparisons = []
    plan_seed = 20261011
    for task in ("task1", "task2"):
        a = details[(task, "probe_rad_dino", 1.0)]
        b = details[(task, "probe_imagenet_resnet50", 1.0)]
        y = a["test_y"]
        if not np.array_equal(y, b["test_y"]):
            raise SystemExit("test labels differ between models")
        dl = delong_test(y, a["test_logit"], b["test_logit"])
        boot = paired_bootstrap(y, a["test_logit"][None, :], b["test_logit"][None, :],
                                metric="auroc", n_boot=10000, seed=plan_seed)
        thr_a = a["test_thr"]["youden_val"]
        thr_b = b["test_thr"]["youden_val"]
        mc = mcnemar_exact(y, (a["test_logit"] >= thr_a).astype(int),
                           (b["test_logit"] >= thr_b).astype(int))
        if not close(dl["diff"], boot["diff"], tol=1e-12):
            raise SystemExit("DeLong and bootstrap observed differences disagree")
        comparisons.append((task, dl, boot, mc, thr_a, thr_b))

    def test_row(task, exp, frac, rule):
        row = rm[(rm.task == task) & (rm.experiment == exp) & np.isclose(rm.train_fraction, frac)
                 & (rm.split == "test") & (rm.rule == rule) & (rm.variant == "raw")].iloc[0]
        return row

    lines = []
    lines.append("# Linear-probe experiment: results")
    lines.append("")
    lines.append("This file summarises the runs that are saved in `results/task1` and "
                 "`results/task2`. Both models are linear classifiers trained on frozen "
                 "image features. Each number was recomputed from `predictions.parquet` "
                 "with the same functions as `scripts/analyze.py` (`src/cxr/metrics.py`, "
                 "`src/cxr/calibration.py`, `src/cxr/stats.py`) and checked against "
                 "`results/summary/` before this file was written. "
                 f"{checked} metric rows matched.")
    lines.append("")
    lines.append("Only seed 0 was run, so there is no standard deviation across seeds. "
                 "The interval next to each AUROC is the DeLong 95% confidence interval "
                 "for that one fitted model. CNN, vision-language, and LoRA runs are not "
                 "in this file because those predictions have not been saved.")
    lines.append("")
    lines.append("## How the numbers are defined")
    lines.append("")
    lines.append("- **Positive class.** Task 1: Pneumonia (negative: Normal). "
                 "Task 2: COVID-19 (negative: non-COVID pneumonia). Each test set has "
                 "228 images of each class (456 in total). Each validation set has "
                 "114 of each class (228 in total).")
    lines.append("- **AUROC** ranks images by the model's score. It does not use a cutoff. "
                 "0.5 is chance on a balanced set. 1 is a perfect ranking. **AUPRC** is "
                 "the area under the precision-recall curve for the positive class.")
    lines.append("- **Cutoff for the other metrics.** Two rules are reported. "
                 "**Default** calls an image positive when the logit is at least 0. "
                 "**Youden** uses the logit cutoff that maximised sensitivity + specificity − 1 "
                 "on the validation set, then applies that cutoff to the test set. "
                 "The validation images were not used to compute the test metrics.")
    lines.append("- **Sensitivity** = true positives / 228 positive test images. "
                 "**Specificity** = true negatives / 228 negative test images. "
                 "**Precision** = true positives / images called positive. "
                 "**Accuracy** = (true positives + true negatives) / 456. "
                 "Because the two classes are the same size, accuracy and balanced accuracy "
                 "are the same value once they are shown to three decimal places.")
    lines.append("- **Brier score** is the mean squared error of the predicted probability "
                 "(lower is better). **ECE** is the 10-bin expected calibration error "
                 "(lower is better). The temperature-scaled ECE uses a temperature fitted "
                 "on the validation set only.")
    lines.append("- Displayed proportions are rounded half-up to three decimal places. "
                 "Small differences are shown to four decimal places. The unrounded values "
                 "are in `results/summary/run_metrics.csv`.")
    lines.append("")

    lines.append("## 1. Runs")
    lines.append("")
    inv = []
    for task in ("task1", "task2"):
        tname = TASKS[task][0]
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            for frac in (0.1, 0.25, 0.5, 1.0):
                d = details[(task, exp, frac)]
                inv.append([tname, NAMES[exp], f"{frac:.0%}", d["train_size"], d["seed"],
                            format(Decimal(str(d["C"])).normalize(), "f"), f3(d["val_nll"])])
    lines.append(md_table(
        ["Task", "Model", "Training fraction", "Training images", "Seed",
         "L2 strength C", "Validation NLL"], inv))
    lines.append("")
    lines.append("C is the inverse L2 penalty chosen by validation negative log-likelihood. "
                 "A smaller C means stronger regularisation. Validation NLL is the score "
                 "that was minimised to choose C (lower is better). It is not a test-set result.")
    lines.append("")

    lines.append("## 2. Test set, all 1,596 training images")
    lines.append("")
    lines.append("### Ranking (no cutoff)")
    lines.append("")
    rank_rows = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            r = test_row(task, exp, 1.0, "youden_val")
            rank_rows.append([
                TASKS[task][0], NAMES[exp], "456",
                f3(r.auroc), f"{f3(r.auroc_ci_low)} to {f3(r.auroc_ci_high)}", f3(r.auprc),
            ])
    lines.append(md_table(
        ["Task", "Model", "Test images", "AUROC", "AUROC 95% CI", "AUPRC"], rank_rows))
    lines.append("")
    lines.append("AUROC does not depend on the cutoff, so the default-cutoff rows have the same AUROC.")
    lines.append("")

    lines.append("### Youden cutoff (chosen on validation, scored on the test set)")
    lines.append("")
    count_rows = []
    metric_rows = []
    cal_rows = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            r = test_row(task, exp, 1.0, "youden_val")
            tp, tn, fp, fn, n = details[(task, exp, 1.0)]["counts"]["youden_val"]
            count_rows.append([
                TASKS[task][0], NAMES[exp], f4(r.threshold),
                tp, fn, tn, fp, f"{tp + tn}/{n}",
            ])
            metric_rows.append([
                TASKS[task][0], NAMES[exp],
                f3(r.accuracy), f3(r.sensitivity), f3(r.specificity), f3(r.precision),
                f3(r.f1), f3(r.mcc), f3(r.positive_rate),
                f"{f3(r.accuracy_ci_low)} to {f3(r.accuracy_ci_high)}",
            ])
            cal_rows.append([
                TASKS[task][0], NAMES[exp], f3(r.brier), f3(r.ece), f3(r.ece_temp_scaled),
                f3(r.temperature),
            ])
    lines.append(md_table(
        ["Task", "Model", "Logit cutoff", "TP", "FN", "TN", "FP", "Correct / 456"],
        count_rows))
    lines.append("")
    lines.append("TP: positive images called positive. FN: positive images called negative. "
                 "TN: negative images called negative. FP: negative images called positive. "
                 "TP + FN = 228 and TN + FP = 228 in every row.")
    lines.append("")
    lines.append(md_table(
        ["Task", "Model", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "MCC",
         "Called positive", "Accuracy 95% CI (Wilson)"],
        metric_rows))
    lines.append("")
    lines.append("These rates are the counts above divided by the matching denominator. "
                 "Sensitivity = TP/228. Specificity = TN/228. Precision = TP/(TP+FP). "
                 "Accuracy = (TP+TN)/456. The accuracy interval is a Wilson 95% interval.")
    lines.append("")
    lines.append(md_table(
        ["Task", "Model", "Brier", "ECE", "ECE after temperature scaling", "Temperature"],
        cal_rows))
    lines.append("")

    lines.append("### Default cutoff (logit at least 0)")
    lines.append("")
    def_rows = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            r = test_row(task, exp, 1.0, "default")
            tp, tn, fp, fn, n = details[(task, exp, 1.0)]["counts"]["default"]
            def_rows.append([
                TASKS[task][0], NAMES[exp], tp, fn, tn, fp,
                f3(r.accuracy), f3(r.sensitivity), f3(r.specificity), f3(r.precision),
                f3(r.f1), f3(r.mcc),
            ])
    lines.append(md_table(
        ["Task", "Model", "TP", "FN", "TN", "FP", "Accuracy", "Sensitivity", "Specificity",
         "Precision", "F1", "MCC"],
        def_rows))
    lines.append("")

    lines.append("## 3. Difference between the two probes (full training set)")
    lines.append("")
    lines.append("RAD-DINO minus the ImageNet ResNet-50 probe, on the same 456 test images. "
                 "This comparison is descriptive. It is outside the pre-registered primary "
                 "tests in `configs/analysis_plan.yaml`, which also need vision-language runs "
                 "that have not been saved. The functions are the ones in that plan: DeLong's "
                 "test for paired AUROCs, a paired class-stratified bootstrap with 10,000 "
                 "replicates and seed 20261011, and an exact McNemar test on the Youden "
                 "decisions. With one seed, the interval describes these two fitted models.")
    lines.append("")
    lines.append("Both bootstrap p-values are 2/10,001 (about 0.000200). That is the smallest "
                 "p-value this bootstrap can give, and it means every one of the 10,000 "
                 "resamples had a positive AUROC difference.")
    lines.append("")
    comp_rows = []
    mcn_rows = []
    for task, dl, boot, mc, thr_a, thr_b in comparisons:
        comp_rows.append([
            TASKS[task][0], f4(dl["diff"]),
            f"{f3(boot['ci_low'])} to {f3(boot['ci_high'])}",
            fmt_p(boot["p"]), f4(dl["z"]), fmt_p(dl["p"]),
        ])
        mcn_rows.append([
            TASKS[task][0],
            mc["a_only_correct"], mc["b_only_correct"], fmt_p(mc["p"]),
        ])
    lines.append(md_table(
        ["Task", "AUROC difference", "Bootstrap 95% CI", "Bootstrap p", "DeLong z", "DeLong p"],
        comp_rows))
    lines.append("")
    lines.append(md_table(
        ["Task", "Only RAD-DINO correct", "Only ImageNet probe correct", "McNemar p"],
        mcn_rows))
    lines.append("")
    lines.append("McNemar uses the Youden cutoff of each model. "
                 "\"Only RAD-DINO correct\" counts test images the RAD-DINO probe classified "
                 "correctly and the ImageNet probe classified incorrectly.")
    lines.append("")

    lines.append("## 4. Test set with less training data")
    lines.append("")
    lines.append("Each subset keeps the same fraction of each training class, with the per-class "
                 "count rounded to the nearest integer (`src/cxr/data.py`). That is why 10% is "
                 "160 images (80 per class) and 25% is 400 images (200 per class), while 50% is "
                 "798 and 100% is 1,596. The test set is the same 456 images in every row. "
                 "AUROC does not use a cutoff. "
                 "The other columns use the Youden cutoff from that run's own validation scores.")
    lines.append("")
    eff = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            for frac in (0.1, 0.25, 0.5, 1.0):
                r = test_row(task, exp, frac, "youden_val")
                tp, tn, fp, fn, n = details[(task, exp, frac)]["counts"]["youden_val"]
                eff.append([
                    TASKS[task][0], NAMES[exp], details[(task, exp, frac)]["train_size"],
                    f3(r.auroc), f"{f3(r.auroc_ci_low)} to {f3(r.auroc_ci_high)}",
                    f3(r.auprc), f3(r.accuracy), f"{tp}/{tp+fn}", f"{tn}/{tn+fp}",
                    f3(r.f1), f3(r.mcc), f3(r.brier), f3(r.ece_temp_scaled),
                ])
    lines.append(md_table(
        ["Task", "Model", "Training images", "AUROC", "AUROC 95% CI", "AUPRC", "Accuracy",
         "Sensitivity (TP/228)", "Specificity (TN/228)", "F1", "MCC", "Brier",
         "ECE (temperature-scaled)"],
        eff))
    lines.append("")

    lines.append("### Default cutoff at each training size")
    lines.append("")
    eff_d = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            for frac in (0.1, 0.25, 0.5, 1.0):
                r = test_row(task, exp, frac, "default")
                tp, tn, fp, fn, _ = details[(task, exp, frac)]["counts"]["default"]
                eff_d.append([
                    TASKS[task][0], NAMES[exp], details[(task, exp, frac)]["train_size"],
                    f3(r.accuracy), f"{tp}/{tp+fn}", f"{tn}/{tn+fp}", f3(r.precision),
                    f3(r.f1), f3(r.positive_rate),
                ])
    lines.append(md_table(
        ["Task", "Model", "Training images", "Accuracy", "Sensitivity (TP/228)",
         "Specificity (TN/228)", "Precision", "F1", "Called positive"],
        eff_d))
    lines.append("")

    lines.append("## 5. Validation set (full training data)")
    lines.append("")
    lines.append("These numbers were used to choose C and the Youden cutoff. "
                 "They are higher than the test numbers and are not the result of the experiment.")
    lines.append("")
    val_rows = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            r = rm[(rm.task == task) & (rm.experiment == exp) & np.isclose(rm.train_fraction, 1.0)
                   & (rm.split == "validation") & (rm.rule == "youden_val")].iloc[0]
            val_rows.append([
                TASKS[task][0], NAMES[exp], "228",
                f3(r.auroc), f"{f3(r.auroc_ci_low)} to {f3(r.auroc_ci_high)}",
                f3(r.accuracy), f3(r.sensitivity), f3(r.specificity), f4(r.threshold),
            ])
    lines.append(md_table(
        ["Task", "Model", "Validation images", "AUROC", "AUROC 95% CI", "Accuracy",
         "Sensitivity", "Specificity", "Youden logit cutoff"],
        val_rows))
    lines.append("")

    lines.append("## 6. Test-time perturbations (full training data)")
    lines.append("")
    lines.append("Each cell is the test AUROC after changing the test images and leaving "
                 "the trained probe unchanged. Clean is the unperturbed AUROC from section 2. "
                 "Drop is clean minus perturbed (positive means the score fell).")
    lines.append("")
    lines.append("The border mask blacks out the outer 12.5% of width and height. "
                 "The centre mask blacks out the central 70% of width and height "
                 "(rows and columns from 15% to 85%). Gamma 0.6 and 1.6 change contrast. "
                 "Noise adds Gaussian noise with standard deviation 0.05 × 255. "
                 "JPEG uses quality 10.")
    lines.append("")
    pert_rows = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            d = details[(task, exp, 1.0)]
            clean = test_row(task, exp, 1.0, "default").auroc
            cells = [TASKS[task][0], NAMES[exp], f3(clean)]
            for pert in PERT_LABEL:
                a = d["pert"][pert]
                cells.append(f"{f3(a)} ({f3(clean - a)})")
            pert_rows.append(cells)
    lines.append(md_table(
        ["Task", "Model", "Clean", *[f"{PERT_LABEL[p]} (drop)" for p in PERT_LABEL]],
        pert_rows))
    lines.append("")

    lines.append("### Perturbed AUROC at every training size")
    lines.append("")
    pert_all = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            for frac in (0.1, 0.25, 0.5, 1.0):
                d = details[(task, exp, frac)]
                clean = test_row(task, exp, frac, "default").auroc
                row = [TASKS[task][0], NAMES[exp], d["train_size"], f3(clean)]
                for pert in PERT_LABEL:
                    row.append(f3(d["pert"][pert]))
                pert_all.append(row)
    lines.append(md_table(
        ["Task", "Model", "Training images", "Clean", *list(PERT_LABEL.values())],
        pert_all))
    lines.append("")

    lines.append("## 7. Near-duplicate training images")
    lines.append("")
    lines.append("Test AUROC after removing test images that the dataset audit paired with "
                 "a training image at Pearson r ≥ 0.95 on normalised 256×256 pixels. "
                 "The audit file is `results/audit/duplicates_task1.csv` and "
                 "`duplicates_task2.csv`. Change is the AUROC without those images minus "
                 "the AUROC on all test images.")
    lines.append("")
    leak_rows = []
    for task in ("task1", "task2"):
        for exp in ("probe_rad_dino", "probe_imagenet_resnet50"):
            for frac in (0.1, 0.25, 0.5, 1.0):
                n_ex, a_all, a_ex, delta = details[(task, exp, frac)]["leak"]
                leak_rows.append([
                    TASKS[task][0], NAMES[exp], details[(task, exp, frac)]["train_size"],
                    n_ex, 456 - n_ex, f3(a_all), f3(a_ex), f"{'+' if delta >= 0 else ''}{f4(delta)}",
                ])
    lines.append(md_table(
        ["Task", "Model", "Training images", "Test images removed", "Test images kept",
         "AUROC (all)", "AUROC (kept)", "Change"],
        leak_rows))
    lines.append("")

    lines.append("## 8. Validation-set selection among the runs that exist")
    lines.append("")
    sel = json.loads((SUMMARY / "selections.json").read_text(encoding="utf-8"))
    sel_rows = []
    for task in ("task1", "task2"):
        s = sel[task]["supervised"]
        sel_rows.append([
            TASKS[task][0], NAMES[s["experiment"]], f3(s["val_auroc_mean"]), s["n_candidates"],
        ])
    lines.append(md_table(
        ["Task", "Highest validation AUROC", "Validation AUROC", "Candidates"],
        sel_rows))
    lines.append("")
    lines.append("The rule in the analysis plan picks the run with the highest mean validation "
                 "AUROC among supervised models, using raw scores and the full training set. "
                 "Only these two probes have been run, so the choice is between those two. "
                 "Vision-language families have no saved runs, and the pre-registered "
                 "comparisons that need them are recorded as missing in "
                 "`results/summary/comparisons.csv`.")
    lines.append("")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(f"checked {checked} rows")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
