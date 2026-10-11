#!/usr/bin/env python
"""Generate every LaTeX table and number macro used by ``paper/main.tex``.

Inputs come only from ``results/`` (audit, power, legacy, summary). Tables
whose inputs do not exist yet are rendered as explicit "pending" placeholders
so that the manuscript never contains hand-typed or stale numbers.

    python scripts/make_tables.py                      # -> paper/generated/
    python scripts/make_tables.py --results /tmp/synth --out /tmp/synth_tables  # pipeline test
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

NAMES = {
    "cnn_resnet50": "ResNet-50 (ImageNet)",
    "cnn_densenet121": "DenseNet-121 (ImageNet)",
    "cnn_xrv_densenet121": "DenseNet-121 (CXR, TorchXRayVision)",
    "probe_rad_dino": "RAD-DINO linear probe",
    "probe_imagenet_resnet50": "ResNet-50 (ImageNet) linear probe",
    "vlm_gemma3_zeroshot": "Gemma-3-4B zero-shot",
    "vlm_medgemma_zeroshot": "MedGemma-4B zero-shot",
    "vlm_gemma3_icl_random": "Gemma-3-4B ICL, random exemplars",
    "vlm_gemma3_icl_knn": "Gemma-3-4B ICL, retrieved exemplars",
    "vlm_medgemma_icl_random": "MedGemma-4B ICL, random exemplars",
    "vlm_medgemma_icl_knn": "MedGemma-4B ICL, retrieved exemplars",
    "vlm_gemma3_lora": "Gemma-3-4B LoRA",
    "vlm_medgemma_lora": "MedGemma-4B LoRA",
    "ensemble_vlm_supervised": "Ensemble (best VLM + best supervised)",
    "cnn_resnet50_data_efficiency": "ResNet-50 (ImageNet)",
    "vlm_gemma3_lora_data_efficiency": "Gemma-3-4B LoRA",
}
MAIN_ORDER = [k for k in NAMES if not k.endswith("data_efficiency")]
TASK_TITLE = {"task1": "Task 1 (Normal vs.\\ Pneumonia)",
              "task2": "Task 2 (Non-COVID pneumonia vs.\\ COVID-19)"}


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def f3(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def pm(mean, sd, n=None):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if n is not None and n > 1 and not (isinstance(sd, float) and math.isnan(sd)):
        return f"{mean:.3f}\\,$\\pm$\\,{sd:.3f}"
    return f"{mean:.3f}"


def pending(ncol, what):
    return (f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Pending: {what} not yet run; generated "
            "automatically once results exist.}}}}\\\\\n")


def write(out, name, body):
    (out / name).write_text(body, encoding="utf-8")


def table(caption, label, colspec, header, rows, note=None, size="\\small"):
    s = (f"\\begin{{table}}[t]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
         f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\\\\\n\\midrule\n{rows}"
         "\\bottomrule\n\\end{tabular}\n")
    if note:
        s += f"\\par\\vspace{{2pt}}\\footnotesize {note}\n"
    return s + "\\end{table}\n"


# ------------------------------------------------------------------ audit/power/legacy

def audit_tables(res, out, macros):
    p = res / "audit" / "audit_summary.json"
    if not p.exists():
        write(out, "tab_audit.tex", "% audit pending\n")
        return
    a = json.loads(p.read_text())
    rows = ""
    for t, v in a["tasks"].items():
        d1 = v["duplicates"]["near_identical_or_exact"]
        d2 = v["duplicates"]["highly_similar_or_closer"]
        ph = v["phash_threshold_check"]
        rows += (f"{TASK_TITLE[t]} & {v['n_images']} & {v['unique_sha256']} & {d1['n_pairs']} & "
                 f"{d1['test_images_with_train_match']} & {d2['n_pairs']} & "
                 f"{d2['label_conflicts']} & {d2['test_images_with_train_match']} & "
                 f"{ph['n_pairs']} ({ph['label_conflicts']}) \\\\\n")
        tag = "One" if t == "task1" else "Two"
        macros[f"Audit{tag}NearIdentical"] = d1["n_pairs"]
        macros[f"Audit{tag}NearIdenticalTest"] = d1["test_images_with_train_match"]
        macros[f"Audit{tag}Similar"] = d2["n_pairs"]
        macros[f"Audit{tag}SimilarConflicts"] = d2["label_conflicts"]
        macros[f"Audit{tag}SimilarTest"] = d2["test_images_with_train_match"]
        macros[f"Audit{tag}PhashPairs"] = ph["n_pairs"]
        macros[f"Audit{tag}PhashConflicts"] = ph["label_conflicts"]
        macros[f"Audit{tag}PhashTrue"] = ph["n_corr64_ge_0.99"]
        macros[f"Audit{tag}Unique"] = v["unique_sha256"]
        sp = v["shortcut_probe"]
        for feat, key in (("thumbnail_8x8", "Thumb"), ("border_only", "Border"),
                          ("geometry", "Geometry"), ("intensity_stats", "Intensity")):
            best = max(sp[feat].values(), key=lambda r: r["validation_auroc"])
            macros[f"Probe{tag}{key}Test"] = f"{best['test_auroc']:.2f}"
            macros[f"Probe{tag}{key}Val"] = f"{best['validation_auroc']:.2f}"
    ct = a.get("cross_task", {})
    macros["CrossTaskShared"] = ct.get("n_task1_images_also_in_task2", "--")
    macros["CrossTaskSplitMismatch"] = ct.get("split_mismatch_pairs", "--")
    write(out, "tab_audit.tex", table(
        "Pixel-level audit of the published splits. Near-identical: Pearson $r\\geq0.99$ "
        "between normalised $256\\times256$ images; highly similar: $r\\geq0.95$. ``Test'' "
        "columns count test images with a matching training image. The last column shows how "
        "many pairs a 64-bit perceptual hash flags at Hamming distance $\\leq4$ (in parentheses: "
        "pairs with different labels), illustrating why hashing alone is unreliable for chest "
        "radiographs.", "tab:audit", "lrrrrrrrr",
        "Task & Images & Unique & \\multicolumn{2}{c}{Near-identical} & "
        "\\multicolumn{3}{c}{Highly similar} & pHash$\\leq$4 \\\\\n"
        " & & & pairs & test & pairs & label conflicts & test & pairs", rows, size="\\scriptsize"))

    rows = ""
    labels = {"geometry": "Image geometry", "border_only": "Border statistics",
              "intensity_stats": "Global intensity statistics",
              "thumbnail_8x8": "$8\\times8$ thumbnail", "mode": "Colour mode"}
    for feat, lab in labels.items():
        cells = []
        for t in ("task1", "task2"):
            r = a["tasks"][t]["shortcut_probe"][feat]
            best = max(r, key=lambda m: r[m]["validation_auroc"])
            cells += [f3(r[best]["validation_auroc"]), f3(r[best]["test_auroc"])]
        rows += f"{lab} & " + " & ".join(cells) + " \\\\\n"
    write(out, "tab_shortcut.tex", table(
        "AUROC of simple classifiers trained on low-level, largely non-anatomical image "
        "features (validation-selected between logistic regression and gradient boosting). "
        "All images are $512\\times512$ RGB, so geometry and colour mode carry no signal.",
        "tab:shortcut", "lcccc",
        "Features & \\multicolumn{2}{c}{Task 1} & \\multicolumn{2}{c}{Task 2}\\\\\n"
        " & val & test & val & test", rows))


def power_tables(res, out, macros):
    p = res / "power" / "power.json"
    if not p.exists():
        return
    pw = json.loads(p.read_text())
    m = pw["n_primary_tests"]
    rows = ""
    for r in pw["auroc"]:
        rows += (f"{r['auroc']:.2f} & {r['ci_half_width']:.3f} & {r['mde_r0.5_alpha_0.05']:.3f} & "
                 f"{r[f'mde_r0.5_holm_worst_case_0.05/{m}']:.3f} & {r['mde_r0.7_alpha_0.05']:.3f}"
                 " \\\\\n")
    write(out, "tab_power.tex", table(
        f"What the fixed test set ($n={pw['n_test']}$; {pw['n_pos']} per class) can resolve. "
        "CI: 95\\% half-width of a single AUROC (Hanley--McNeil). MDE: minimum detectable "
        "difference between two correlated AUROCs at 80\\% power, for inter-model score "
        f"correlation $r$, at $\\alpha=0.05$ and at the Holm worst case $\\alpha/{m}$.",
        "tab:power", "ccccc",
        "AUROC & CI half-width & MDE ($r{=}0.5$) & MDE ($r{=}0.5$, $\\alpha/" + str(m) +
        "$) & MDE ($r{=}0.7$)", rows))
    for r in pw["accuracy"]:
        if abs(r["metric_value"] - 0.85) < 1e-9:
            macros["PowerAccHalfWidth"] = f"{100 * r['ci_half_width']:.1f}"
            macros["PowerClassHalfWidth"] = f"{100 * r['per_class_ci_half_width']:.1f}"
    for r in pw["auroc"]:
        if abs(r["auroc"] - 0.85) < 1e-9:
            macros["PowerAucMde"] = f"{r['mde_r0.5_alpha_0.05']:.3f}"
            macros["PowerAucHalfWidth"] = f"{r['ci_half_width']:.3f}"
    for r in pw["mcnemar"]:
        if abs(r["discordant_rate"] - 0.15) < 1e-9:
            macros["PowerMcnemarMde"] = f"{100 * r['mde_alpha_0.05']:.1f}"
    macros["NPrimaryTests"] = m


def legacy_table(res, out):
    p = res / "legacy" / "legacy_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fam = {"cnn": "ResNet-50", "finetune": "Gemma-3-4B LoRA", "icl_base": "Gemma-3-4B zero-shot",
           "icl_nshot": "Gemma-3-4B ICL"}
    rows = ""
    for t in ("task1", "task2"):
        d = df[df.task == t]
        for _, r in d.iterrows():
            shots = "" if pd.isna(r.shots) else f", {int(r.shots)} shots"
            ver = "v1.0.0" if "1_0_0" in r.notebook else ("v1.0.2" if "1_0_2" in r.notebook
                                                         else "orig.")
            rows += (f"{'1' if t == 'task1' else '2'} & {fam[r.family]}{shots} & {ver} & "
                     f"{r.accuracy_pct:.2f} & {f3(r.get('recall_negative'))} & "
                     f"{f3(r.get('recall_positive'))} \\\\\n")
    write(out, "tab_legacy.tex", table(
        "Preliminary single-run results from the original notebooks, parsed automatically "
        "from their stored outputs (\\texttt{scripts/extract\\_legacy\\_results.py}). These "
        "runs predate the protocol of this paper and have known issues (Section~\\ref{sec:"
        "legacy}); they are reported for transparency, not as findings.", "tab:legacy",
        "clcccc", "Task & Method & Notebook & Acc.\\ (\\%) & Recall$_{neg}$ & Recall$_{pos}$",
        rows, size="\\scriptsize"))


# ---------------------------------------------------------------------- results

def load_summary(res):
    s = res / "summary"
    if not (s / "condition_summary.csv").exists():
        return None
    out = {}
    for name in ("condition_summary", "comparisons", "robustness", "generation_vs_likelihood",
                 "leakage_sensitivity"):
        f = s / f"{name}.csv"
        try:
            out[name] = pd.read_csv(f) if f.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            out[name] = pd.DataFrame()
    out["selections"] = json.loads((s / "selections.json").read_text())
    return out


def best_shots(cs, task, exp):
    v = cs[(cs.task == task) & (cs.experiment == exp) & (cs.split == "validation") &
           (cs.variant == "raw") & (cs.rule == "default") & (cs.train_fraction == 1.0)]
    if v.empty:
        return None
    return int(v.sort_values(["auroc_mean", "shots"], ascending=[False, True]).iloc[0].shots)


def cond_row(cs, task, exp, shots, variant, rule, split="test", frac=1.0):
    c = cs[(cs.task == task) & (cs.experiment == exp) & (cs.split == split) &
           (cs.variant == variant) & (cs.rule == rule) & (cs.train_fraction == frac) &
           (cs.shots == shots)]
    return None if c.empty else c.iloc[0]


def main_tables(S, out, macros):
    ncol = 8
    header = ("Method & $k$ & AUROC & Bal.\\ acc. & Sens. & Spec. & ECE & runs")
    note = ("Test-set mean\\,$\\pm$\\,SD over seeds (supervised, LoRA) or exemplar draws (ICL). "
            "AUROC uses raw scores. Threshold metrics use the decision threshold maximising "
            "Youden's $J$ on validation; ECE is after validation temperature scaling. $k$: "
            "number of in-context examples, chosen on validation.")
    for t in ("task1", "task2"):
        rows = ""
        if S is None:
            rows = pending(ncol, "experiments")
        else:
            cs = S["condition_summary"]
            for exp in MAIN_ORDER:
                shots = best_shots(cs, t, exp)
                if shots is None:
                    continue
                r = cond_row(cs, t, exp, shots, "raw", "youden_val")
                if r is None:
                    continue
                k = "--" if shots < 0 else str(shots)
                n = int(r.n_runs)
                rows += (f"{NAMES[exp]} & {k} & {pm(r.auroc_mean, r.auroc_std, n)} & "
                         f"{pm(r.balanced_accuracy_mean, r.balanced_accuracy_std, n)} & "
                         f"{f3(r.sensitivity_mean)} & {f3(r.specificity_mean)} & "
                         f"{f3(r.ece_temp_scaled_mean)} & {n} \\\\\n")
            rows = rows or pending(ncol, "final (test-set) runs")
        write(out, f"tab_main_{t}.tex", table(f"Main results, {TASK_TITLE[t]}.",
                                              f"tab:main_{t}", "lccccccc", header, rows, note,
                                              size="\\scriptsize"))


def icl_table(S, out):
    exps = ["vlm_gemma3_icl_random", "vlm_gemma3_icl_knn", "vlm_medgemma_icl_random",
            "vlm_medgemma_icl_knn"]
    header = ("Task & Method & $k$ & AUROC & Bal.\\ acc.\\ (raw) & Bal.\\ acc.\\ "
              "(val-centred) & Bal.\\ acc.\\ (contextual) & Pos.\\ rate (raw)")
    rows = ""
    if S is not None:
        cs = S["condition_summary"]
        for t in ("task1", "task2"):
            for exp in ["vlm_gemma3_zeroshot", "vlm_medgemma_zeroshot"] + exps:
                c = cs[(cs.task == t) & (cs.experiment == exp) & (cs.split == "test")]
                for shots in sorted(c.shots.unique()):
                    raw = cond_row(cs, t, exp, shots, "raw", "default")
                    vc = cond_row(cs, t, exp, shots, "val_centered", "default")
                    cc = cond_row(cs, t, exp, shots, "contextual", "default")
                    if raw is None:
                        continue
                    n = int(raw.n_runs)
                    rows += (f"{t[-1]} & {NAMES[exp]} & {0 if shots < 0 else shots} & "
                             f"{pm(raw.auroc_mean, raw.auroc_std, n)} & "
                             f"{pm(raw.balanced_accuracy_mean, raw.balanced_accuracy_std, n)} & "
                             f"{'--' if vc is None else pm(vc.balanced_accuracy_mean, vc.balanced_accuracy_std, n)} & "  # noqa: E501
                             f"{'--' if cc is None else pm(cc.balanced_accuracy_mean, cc.balanced_accuracy_std, n)} & "  # noqa: E501
                             f"{f3(raw.positive_rate_mean)} \\\\\n")
    write(out, "tab_icl.tex", table(
        "Zero-shot and in-context learning on the test set as a function of the number of "
        "examples $k$ (mean\\,$\\pm$\\,SD over 5 draws). Balanced accuracy uses the argmax "
        "decision ($\\text{logit}\\geq0$) on raw, validation-centred and contextually "
        "calibrated scores. ``Pos.\\ rate'' is the fraction predicted positive; values near 0 "
        "or 1 indicate collapse onto one label.", "tab:icl", "clcccccc", header,
        rows or pending(8, "ICL experiments"), size="\\scriptsize"))


def comparison_table(S, out):
    header = ("Tier & Task & Comparison (A vs.\\ B) & Metric & $\\Delta$ (95\\% CI) & $p$ & "
              "$p_{\\text{Holm}}$")
    rows = ""
    if S is not None and not S["comparisons"].empty:
        c = S["comparisons"]
        for _, r in c.iterrows():
            if r.status != "ok":
                rows += (f"{r.tier} & {r.task[-1]} & {esc(r.id)} & {r.metric} & "
                         f"\\multicolumn{{3}}{{l}}{{{esc(r.status)}}} \\\\\n")
                continue
            holm = "--" if pd.isna(r.get("p_holm")) else f"{r.p_holm:.3g}"
            rows += (f"{r.tier} & {r.task[-1]} & {esc(r.id)} & {esc(r.metric)} & "
                     f"{r.boot_diff:+.3f} [{r.boot_ci_low:+.3f}, {r.boot_ci_high:+.3f}] & "
                     f"{r.boot_p:.3g} & {holm} \\\\\n")
    write(out, "tab_comparisons.tex", table(
        "Pre-registered comparisons on the test set. $\\Delta$ is the difference in the "
        "seed-averaged metric (A$-$B) with a paired, class-stratified bootstrap CI over test "
        "cases (10{,}000 replicates). Conditions are selected on validation only. Primary "
        "$p$-values are Holm-adjusted across both tasks; secondary comparisons are exploratory.",
        "tab:comparisons", "llp{5.2cm}lccc", header, rows or pending(7, "final runs"),
        size="\\scriptsize"))


def robustness_table(S, out):
    perts = ["border_mask", "centre_mask", "gamma_low", "gamma_high", "noise", "jpeg"]
    header = "Task & Method & Clean & " + " & ".join(esc(p) for p in perts)
    rows = ""
    if S is not None and not S["robustness"].empty:
        rb = S["robustness"]
        cs = S["condition_summary"]
        for t in ("task1", "task2"):
            for exp in MAIN_ORDER:
                d = rb[(rb.task == t) & (rb.experiment == exp) & (rb.train_fraction == 1.0)]
                if d.empty:
                    continue
                clean = cond_row(cs, t, exp, int(d.shots.iloc[0]), "raw", "default")
                cells = []
                for p in perts:
                    x = d[d.perturbation == p]
                    cells.append("--" if x.empty else f"{x.auroc_mean.iloc[0]:.3f}")
                rows += (f"{t[-1]} & {NAMES[exp]} & {f3(None if clean is None else clean.auroc_mean)}"
                         f" & " + " & ".join(cells) + " \\\\\n")
    write(out, "tab_robustness.tex", table(
        "Test AUROC (mean over seeds) under test-time perturbations. ``centre\\_mask'' removes "
        "the central 70\\% of the image (most of the lung fields); AUROC well above 0.5 after "
        "this perturbation indicates reliance on non-pulmonary cues.", "tab:robustness",
        "cl" + "c" * (len(perts) + 1), header, rows or pending(len(perts) + 3, "final runs"),
        size="\\scriptsize"))


def efficiency_table(S, out):
    header = "Task & Method & 10\\% & 25\\% & 50\\% & 100\\%"
    rows = ""
    if S is not None:
        cs = S["condition_summary"]
        pairs = [("cnn_resnet50", "cnn_resnet50_data_efficiency"),
                 ("vlm_gemma3_lora", "vlm_gemma3_lora_data_efficiency"),
                 ("probe_rad_dino", "probe_rad_dino")]
        for t in ("task1", "task2"):
            for full, part in pairs:
                cells = []
                for frac in (0.1, 0.25, 0.5, 1.0):
                    r = cond_row(cs, t, full if frac == 1.0 else part, -1, "raw", "default",
                                 frac=frac)
                    cells.append("--" if r is None else pm(r.auroc_mean, r.auroc_std,
                                                           int(r.n_runs)))
                if any(c != "--" for c in cells):
                    rows += f"{t[-1]} & {NAMES[full]} & " + " & ".join(cells) + " \\\\\n"
    write(out, "tab_efficiency.tex", table(
        "Test AUROC versus the fraction of the 1{,}596 training images used (class-stratified "
        "subsets; mean\\,$\\pm$\\,SD over seeds).", "tab:efficiency", "clcccc", header,
        rows or pending(6, "data-efficiency experiments"), size="\\scriptsize"))


def generation_table(S, out):
    header = ("Task & Method & Parse failures & Acc.\\ (generation) & Acc.\\ (likelihood) & "
              "Agreement")
    rows = ""
    if S is not None and not S["generation_vs_likelihood"].empty:
        g = S["generation_vs_likelihood"]
        g = g[g.split == "test"]
        agg = g.groupby(["task", "experiment"]).mean(numeric_only=True).reset_index()
        for _, r in agg.iterrows():
            rows += (f"{r.task[-1]} & {NAMES.get(r.experiment, esc(r.experiment))} & "
                     f"{f3(r.parse_failure_rate)} & {f3(r.accuracy_generation)} & "
                     f"{f3(r.accuracy_likelihood)} & {f3(r.agreement)} \\\\\n")
    write(out, "tab_generation.tex", table(
        "Original protocol (greedy generation + string matching) versus label-likelihood "
        "scoring with the argmax rule, on the same test images (mean over seeds).",
        "tab:generation", "clcccc", header, rows or pending(6, "final VLM runs"),
        size="\\scriptsize"))


def leakage_table(S, out):
    header = "Task & Method & $n$ excluded & AUROC (all) & AUROC (excl.) & max $|\\Delta|$"
    rows = ""
    if S is not None and not S["leakage_sensitivity"].empty:
        d = S["leakage_sensitivity"]
        d = d[(d.train_fraction == 1.0)]
        cs = S["condition_summary"]
        for t in ("task1", "task2"):
            for exp in MAIN_ORDER:
                shots = best_shots(cs, t, exp) if not cs.empty else None
                x = d[(d.task == t) & (d.experiment == exp)]
                if shots is not None:
                    x = x[x.shots == shots]
                if x.empty:
                    continue
                r = x.iloc[0]
                rows += (f"{t[-1]} & {NAMES[exp]} & {int(r.n_excluded)} & {f3(r.auroc_all)} & "
                         f"{f3(r.auroc_excluding_similar)} & {f3(r.change_max_abs)} \\\\\n")
    write(out, "tab_leakage.tex", table(
        "Leakage sensitivity: test AUROC (mean over runs) with and without the test images that "
        "have a highly similar ($r\\geq0.95$) training image.", "tab:leakage", "clcccc", header,
        rows or pending(6, "final runs"), size="\\scriptsize"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated"))
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    if (res / "SYNTHETIC").exists() and out.resolve().is_relative_to((ROOT / "paper").resolve()):
        sys.exit("refusing to write tables from SYNTHETIC results into paper/")
    out.mkdir(parents=True, exist_ok=True)
    macros = {}
    audit_tables(res, out, macros)
    power_tables(res, out, macros)
    legacy_table(res, out)
    S = load_summary(res)
    main_tables(S, out, macros)
    icl_table(S, out)
    comparison_table(S, out)
    robustness_table(S, out)
    efficiency_table(S, out)
    generation_table(S, out)
    leakage_table(S, out)
    macros["ResultsAvailable"] = "yes" if S is not None and not S["condition_summary"].empty \
        else "no"
    lines = ["% Auto-generated by scripts/make_tables.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    write(out, "numbers.tex", "\n".join(lines) + "\n")
    print(f"wrote {len(list(out.glob('*.tex')))} files to {out}")


if __name__ == "__main__":
    main()
