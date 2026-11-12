#!/usr/bin/env python
"""Leakage and shortcut audit of both POLCOVID tasks (CPU only).

Writes to ``results/audit/``:
  records_<task>.parquet      per-image hashes and features
  duplicates_<task>.csv       graded duplicate / highly-similar pairs within a task
  cross_task_duplicates.csv   graded pairs between task 1 and task 2 images
  class_summary_<task>.csv    per split/class medians of geometry and intensity
  top_pairs_<task>.png        contact sheet of the most similar cross-split pairs
  audit_summary.json          all headline numbers used in the paper
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr import audit  # noqa: E402
from cxr.data import load_splits  # noqa: E402
from cxr.tasks import TASKS  # noqa: E402


def contact_sheet(pairs, ds, path, n=8):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = pairs[pairs.split_a != pairs.split_b].head(n)
    if p.empty:
        return
    fig, axes = plt.subplots(2, len(p), figsize=(2.0 * len(p), 4.4))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for k, r in enumerate(p.itertuples()):
        for row, (s, i, lab) in enumerate(((r.split_a, r.index_a, r.label_a),
                                           (r.split_b, r.index_b, r.label_b))):
            ax = axes[row, k]
            ax.imshow(ds[s][int(i)]["image"].convert("L"), cmap="gray")
            ax.set_title(f"{s}[{i}] y={lab}", fontsize=7)
            ax.axis("off")
        axes[1, k].text(0.5, -0.12, f"r={r.corr256:.3f}", transform=axes[1, k].transAxes,
                        ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "audit"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"method": {
        "candidate": f"Pearson r of normalised {audit.DUP_THUMB}x{audit.DUP_THUMB} thumbnails "
                     f">= {audit.HIGHLY_SIMILAR - 0.03:.2f} (exhaustive, all pairs)",
        "verification": f"Pearson r at {audit.VERIFY_SIZE}x{audit.VERIFY_SIZE}",
        "near_identical": audit.NEAR_IDENTICAL, "highly_similar": audit.HIGHLY_SIMILAR},
        "tasks": {}}
    store = {}
    for name, task in TASKS.items():
        print(f"[{name}] {task.dataset_id}@{task.revision}", flush=True)
        ds = load_splits(task)
        records, thumbs, dthumbs = audit.compute_records(ds)
        records.to_parquet(out / f"records_{name}.parquet", index=False)
        i, j, c = audit.candidate_pairs(dthumbs)
        print(f"  {len(i)} candidate pairs", flush=True)
        pairs = audit.verified_pairs(records, ds, i, j, coarse=c)
        pairs.to_csv(out / f"duplicates_{name}.csv", index=False)
        contact_sheet(pairs, ds, out / f"top_pairs_{name}.png")
        audit.class_summary(records).to_csv(out / f"class_summary_{name}.csv", index=False)
        summary["tasks"][name] = {
            "dataset_id": task.dataset_id, "revision": task.revision,
            "class_names": list(task.class_names),
            "split_sizes": records.split.value_counts().to_dict(),
            "positive_fraction": records.groupby("split").label.mean().to_dict(),
            "n_images": int(len(records)), "unique_sha256": int(records.sha256.nunique()),
            "n_candidates": int(len(i)),
            "duplicates": audit.summarise_duplicates(pairs),
            "phash_threshold_check": audit.phash_threshold_check(records, dthumbs),
            "modes_by_label": {str(k): v for k, v in records.groupby("label")["mode"]
                               .value_counts().unstack(fill_value=0).to_dict("index").items()},
            "shortcut_probe": audit.shortcut_probe(records, thumbs),
        }
        print(json.dumps(summary["tasks"][name]["duplicates"], indent=1), flush=True)
        store[name] = (records, dthumbs, ds)

    (ra, ta, da), (rb, tb, db) = store["task1"], store["task2"]
    i, j, c = audit.candidate_pairs(ta, tb)
    cross = audit.verified_pairs(ra, da, i, j, rec_b=rb, ds_b=db, coarse=c)
    cross.to_csv(out / "cross_task_duplicates.csv", index=False)
    near = cross[cross.grade.isin(["exact", "near_identical"])]
    summary["cross_task"] = {
        "n_task1_images_also_in_task2": int(near.drop_duplicates(["split_a", "index_a"]).shape[0]),
        "task1_label_of_shared": {str(k): int(v) for k, v in
                                  near.drop_duplicates(["split_a", "index_a"]).label_a
                                  .value_counts().items()},
        "split_mismatch_pairs": int((near.split_a != near.split_b).sum()),
        "split_mismatch_detail": {f"{a}->{b}": int(v) for (a, b), v in
                                  near.groupby(["split_a", "split_b"]).size().items() if a != b},
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["cross_task"], indent=1))


if __name__ == "__main__":
    main()
