#!/usr/bin/env python
"""Generate paper figures from ``results/``; figures without inputs are skipped.

The manuscript includes each figure through ``\\IfFileExists`` so missing
figures render as a labelled placeholder rather than failing the build.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_tables import NAMES, TASK_TITLE, best_shots  # noqa: E402

from cxr.io import load_runs  # noqa: E402
from cxr.metrics import sigmoid  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "figure.dpi": 150})


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def icl_curves(cs, out):
    exps = [e for e in ("vlm_gemma3_icl_random", "vlm_gemma3_icl_knn", "vlm_medgemma_icl_random",
                        "vlm_medgemma_icl_knn") if e in set(cs.experiment)]
    if not exps:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.8), sharex=True)
    for col, task in enumerate(("task1", "task2")):
        for row, (metric, variant) in enumerate((("auroc", "raw"),
                                                 ("balanced_accuracy", None))):
            ax = axes[row, col]
            for ci, exp in enumerate(exps):
                variants = ["raw"] if variant else ["raw", "val_centered"]
                for var in variants:
                    d = cs[(cs.task == task) & (cs.experiment == exp) & (cs.split == "test") &
                           (cs.variant == var) & (cs.rule == "default")].sort_values("shots")
                    if d.empty:
                        continue
                    ls = "-" if var == "raw" else "--"
                    ax.errorbar(d.shots, d[f"{metric}_mean"], yerr=d[f"{metric}_std"],
                                ls=ls, marker="o", ms=3, capsize=2, color=f"C{ci}",
                                label=f"{NAMES[exp]}" + ("" if var == "raw" else " (val-centred)"))
            for zs, c in (("vlm_gemma3_zeroshot", "0.4"), ("vlm_medgemma_zeroshot", "0.7")):
                z = cs[(cs.task == task) & (cs.experiment == zs) & (cs.split == "test") &
                       (cs.variant == "raw") & (cs.rule == "default")]
                if not z.empty:
                    ax.axhline(z[f"{metric}_mean"].iloc[0], color=c, lw=0.8, ls=":",
                               label=NAMES[zs])
            ax.axhline(0.5, color="k", lw=0.5)
            ax.set_ylabel("AUROC" if metric == "auroc" else "Balanced accuracy (argmax)")
            if row == 0:
                ax.set_title(TASK_TITLE[task].replace("\\ ", " "))
            if row == 1:
                ax.set_xlabel("in-context examples $k$")
    axes[0, 0].legend(loc="lower right", ncol=1)
    from matplotlib.lines import Line2D

    axes[1, 0].legend(handles=[Line2D([], [], color="k", ls="-", label="raw scores"),
                               Line2D([], [], color="k", ls="--", label="validation-centred")],
                      loc="lower right")
    save(fig, out, "fig_icl_shots")


def roc_and_reliability(runs, sel, out):
    from sklearn.metrics import roc_curve

    fams = [("supervised", "C0"), ("vlm_any", "C3"), ("gemma3_zeroshot", "C7")]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6))
    any_plot = False
    for col, task in enumerate(("task1", "task2")):
        for fam, color in fams:
            c = sel.get(task, {}).get(fam)
            if not c:
                continue
            rs = [r for r in runs if r["task"] == task and r["experiment"] == c["experiment"]
                  and r["meta"].get("final") and (r["meta"].get("shots") or -1) == c["shots"]
                  and float(r["meta"].get("train_fraction", 1)) == 1.0]
            if not rs:
                continue
            any_plot = True
            p = rs[0]["predictions"]
            t = p[(p.split == "test") & (p.perturbation == "none")]
            fpr, tpr, _ = roc_curve(t.y_true, t.logit)
            axes[0, col].plot(fpr, tpr, color=color, lw=1,
                              label=f"{NAMES.get(c['experiment'], c['experiment'])}")
            prob = sigmoid(t.logit.to_numpy())
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(prob, bins[1:-1]), 0, 9)
            xs = [prob[idx == b].mean() for b in range(10) if (idx == b).any()]
            ys = [t.y_true.to_numpy()[idx == b].mean() for b in range(10) if (idx == b).any()]
            axes[1, col].plot(xs, ys, marker="o", ms=3, color=color, lw=1)
        axes[0, col].plot([0, 1], [0, 1], "k:", lw=0.5)
        axes[0, col].set(xlabel="1 - specificity", ylabel="sensitivity",
                         title=TASK_TITLE[task].replace("\\ ", " "))
        axes[0, col].legend(loc="lower right")
        axes[1, col].plot([0, 1], [0, 1], "k:", lw=0.5)
        axes[1, col].set(xlabel="predicted probability (uncalibrated)",
                         ylabel="observed frequency")
    if any_plot:
        save(fig, out, "fig_roc_reliability")
    else:
        plt.close(fig)


def efficiency(cs, out):
    pairs = [("cnn_resnet50", "cnn_resnet50_data_efficiency"),
             ("vlm_gemma3_lora", "vlm_gemma3_lora_data_efficiency"),
             ("probe_rad_dino", "probe_rad_dino")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    drew = False
    for col, task in enumerate(("task1", "task2")):
        ax = axes[col]
        for full, part in pairs:
            xs, ms, sds = [], [], []
            for frac in (0.1, 0.25, 0.5, 1.0):
                d = cs[(cs.task == task) & (cs.experiment == (full if frac == 1.0 else part)) &
                       (cs.split == "test") & (cs.variant == "raw") & (cs.rule == "default") &
                       (cs.train_fraction == frac)]
                if not d.empty:
                    xs.append(frac * 1596)
                    ms.append(d.auroc_mean.iloc[0])
                    sds.append(0 if pd.isna(d.auroc_std.iloc[0]) else d.auroc_std.iloc[0])
            if len(xs) > 1:
                drew = True
                ax.errorbar(xs, ms, yerr=sds, marker="o", ms=3, capsize=2, label=NAMES[full])
        ax.set_xscale("log")
        ax.set(xlabel="training images", title=TASK_TITLE[task].replace("\\ ", " "))
    axes[0].set_ylabel("test AUROC")
    axes[0].legend()
    if drew:
        save(fig, out, "fig_efficiency")
    else:
        plt.close(fig)


def robustness(rb, cs, out):
    if rb.empty:
        return
    perts = ["border_mask", "centre_mask", "gamma_low", "gamma_high", "noise", "jpeg"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for col, task in enumerate(("task1", "task2")):
        d = rb[(rb.task == task) & (rb.train_fraction == 1.0)]
        exps = [e for e in NAMES if e in set(d.experiment)]
        w = 0.8 / max(1, len(exps))
        for k, exp in enumerate(exps):
            x = d[(d.experiment == exp) & (d.shots == best_shots(cs, task, exp))]
            x = x.set_index("perturbation").reindex(perts)
            axes[col].bar(np.arange(len(perts)) + k * w, x.auroc_mean, w, yerr=x.auroc_std,
                          label=NAMES[exp], capsize=1)
        axes[col].axhline(0.5, color="k", lw=0.5)
        axes[col].set_xticks(np.arange(len(perts)) + 0.4 - w / 2,
                             [p.replace("_", "\n") for p in perts])
        axes[col].set_title(TASK_TITLE[task].replace("\\ ", " "))
    axes[0].set_ylabel("test AUROC under perturbation")
    axes[1].legend(fontsize=5.5, loc="lower left")
    save(fig, out, "fig_robustness")


def audit_figs(res, out):
    for t in ("task1", "task2"):
        src = res / "audit" / f"top_pairs_{t}.png"
        if src.exists():
            shutil.copy(src, out / f"fig_audit_pairs_{t}.png")
    gc = res / "gradcam"
    if gc.exists():
        for f in gc.glob("*.png"):
            shutil.copy(f, out / f"fig_{f.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    if (res / "SYNTHETIC").exists() and out.resolve().is_relative_to((ROOT / "paper").resolve()):
        sys.exit("refusing to write figures from SYNTHETIC results into paper/")
    out.mkdir(parents=True, exist_ok=True)
    audit_figs(res, out)
    s = res / "summary"
    if (s / "condition_summary.csv").exists():
        cs = pd.read_csv(s / "condition_summary.csv")
        if not cs.empty:
            icl_curves(cs, out)
            efficiency(cs, out)
            sel = json.loads((s / "selections.json").read_text())
            runs = [r for r in load_runs(res) if r["task"] in ("task1", "task2")]
            roc_and_reliability(runs, sel, out)
            try:
                robustness(pd.read_csv(s / "robustness.csv"), cs, out)
            except pd.errors.EmptyDataError:
                pass
    print("figures:", sorted(p.name for p in out.glob("*")))


if __name__ == "__main__":
    main()
