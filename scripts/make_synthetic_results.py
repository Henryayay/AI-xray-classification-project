#!/usr/bin/env python
"""Write SYNTHETIC prediction files with the real layout, for pipeline tests only.

The output directory is marked with a ``SYNTHETIC`` file, and table/figure
generation refuses to write into ``paper/`` from a results root that has it.
Never use these numbers for anything but testing the code.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.io import write_meta, write_predictions  # noqa: E402

EXPERIMENTS = {
    "cnn_resnet50": ("cnn", 1.6, [None], [1.0]),
    "cnn_densenet121": ("cnn", 1.5, [None], [1.0]),
    "cnn_resnet50_data_efficiency": ("cnn", 1.0, [None], [0.1, 0.25, 0.5]),
    "probe_rad_dino": ("linear_probe", 1.4, [None], [0.1, 0.25, 0.5, 1.0]),
    "vlm_gemma3_zeroshot": ("vlm_zeroshot", 0.3, [None], [1.0]),
    "vlm_medgemma_zeroshot": ("vlm_zeroshot", 0.7, [None], [1.0]),
    "vlm_gemma3_icl_random": ("vlm_icl", 0.6, [2, 4, 6, 8, 10], [1.0]),
    "vlm_gemma3_icl_knn": ("vlm_icl", 0.9, [2, 4, 6, 8, 10], [1.0]),
    "vlm_medgemma_icl_random": ("vlm_icl", 0.8, [2, 4], [1.0]),
    "vlm_medgemma_icl_knn": ("vlm_icl", 1.0, [2, 4], [1.0]),
    "vlm_gemma3_lora": ("vlm_lora", 1.9, [None], [1.0]),
    "vlm_medgemma_lora": ("vlm_lora", 2.0, [None], [1.0]),
}
PERTS = ["border_mask", "centre_mask", "noise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=120)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SYNTHETIC").write_text("synthetic data for pipeline tests; not experimental results\n")
    rng = np.random.default_rng(0)
    for task in ("task1", "task2"):
        sizes = {"validation": args.n_val, "test": args.n_test}
        ys = {s: np.repeat([0, 1], n // 2) for s, n in sizes.items()}
        for exp, (kind, sep, shots_list, fracs) in EXPERIMENTS.items():
            for seed in range(args.seeds):
                for shots in shots_list:
                    for frac in fracs:
                        bias = -1.5 if kind.startswith("vlm") and kind != "vlm_lora" else 0.0
                        frames = []
                        for s, y in ys.items():
                            perts = ["none"] + (PERTS if s == "test" else [])
                            for p in perts:
                                eff = sep * frac ** 0.3 * (0.5 if p == "centre_mask" else 1.0)
                                z = rng.normal(0, 1, len(y)) + eff * (2 * y - 1) + bias
                                f = pd.DataFrame({"split": s, "index": np.arange(len(y)),
                                                  "y_true": y, "logit": z, "perturbation": p})
                                if kind.startswith("vlm"):
                                    f["logit_cf"] = bias + rng.normal(0, 0.1)
                                    f["parsed_label"] = np.where(rng.random(len(y)) < 0.05, -1,
                                                                 (z >= 0).astype(int))
                                frames.append(f)
                        name = f"seed{seed}" + (f"_frac{frac:g}" if frac != 1.0 else "") + (
                            f"_shots{shots}" if shots else "")
                        d = out / task / exp / name
                        d.mkdir(parents=True, exist_ok=True)
                        write_predictions(d, pd.concat(frames))
                        write_meta(d, {"task": task, "experiment": exp, "run": name,
                                       "kind": kind, "seed": seed, "train_fraction": frac,
                                       "shots": shots, "final": True, "limit": 0,
                                       "synthetic": True})
    print(json.dumps({"written": str(out)}))


if __name__ == "__main__":
    main()
