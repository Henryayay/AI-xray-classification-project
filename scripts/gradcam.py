#!/usr/bin/env python
"""Grad-CAM for trained CNN checkpoints: example panels and a quantitative localisation score.

For every test image the fraction of Grad-CAM mass inside the central region
masked by the ``centre_mask`` perturbation (rows/cols 15%-85%, approximately
the lung fields) is computed. Uniform attention gives 0.49; values well below
that mean the model attends to the periphery (labels, markers, collimation).

    python scripts/gradcam.py --task task1 --experiment cnn_resnet50 --run seed0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.cnn import GradCAM, build_model, last_conv_layer, make_transform  # noqa: E402
from cxr.data import load_splits, pad_to_square, to_rgb  # noqa: E402
from cxr.tasks import get_task  # noqa: E402

CENTRE = (0.15, 0.85)
UNIFORM_CENTRE_MASS = (CENTRE[1] - CENTRE[0]) ** 2


def centre_mass(cam):
    h, w = cam.shape
    r0, r1 = int(CENTRE[0] * h), int(CENTRE[1] * h)
    c0, c1 = int(CENTRE[0] * w), int(CENTRE[1] * w)
    tot = cam.sum() + 1e-12
    return float(cam[r0:r1, c0:c1].sum() / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--run", default="seed0")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(args.results) / args.task / args.experiment / args.run
    meta = json.loads((run / "meta.json").read_text())
    cfg = meta["config"]
    if not meta.get("final"):
        sys.exit("Grad-CAM uses test images; run the experiment with --final first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["arch"], pretrained=False).to(device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    cam_fn = GradCAM(model, last_conv_layer(model))
    tf = make_transform(cfg["arch"], cfg.get("image_size", 224), train=False)
    test = load_splits(get_task(args.task))["test"]
    n = args.limit or len(test)

    masses, logits, ys, cams = [], [], [], {}
    for i in range(n):
        ex = test[i]
        x = tf(ex["image"])[None].to(device)
        cam, lg = cam_fn(x)
        cam = torch.nn.functional.interpolate(torch.from_numpy(cam)[None], size=x.shape[-2:],
                                              mode="bilinear", align_corners=False)[0, 0].numpy()
        masses.append(centre_mass(cam))
        logits.append(float(lg[0]))
        ys.append(int(ex["label"]))
        if len(cams) < 200:
            cams[i] = cam
    masses, logits, ys = map(np.asarray, (masses, logits, ys))
    out = Path(args.results) / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"task": args.task, "experiment": args.experiment, "run": args.run, "n": int(n),
               "uniform_centre_mass": UNIFORM_CENTRE_MASS,
               "centre_mass_mean": float(masses.mean()),
               "centre_mass_median": float(np.median(masses)),
               "centre_mass_by_class": {int(c): float(masses[ys == c].mean()) for c in (0, 1)},
               "frac_images_centre_mass_below_uniform": float(np.mean(masses <
                                                                      UNIFORM_CENTRE_MASS))}
    tag = f"{args.task}_{args.experiment}_{args.run}"
    (out / f"gradcam_{tag}.json").write_text(json.dumps(summary, indent=2))

    correct = (logits >= 0).astype(int) == ys
    pick = []
    for c in (0, 1):
        for ok in (True, False):
            cand = [i for i in cams if ys[i] == c and correct[i] == ok]
            pick += cand[: max(1, args.n_show // 4)]
    fig, axes = plt.subplots(2, len(pick), figsize=(1.6 * len(pick), 3.4))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for k, i in enumerate(pick):
        img = np.asarray(pad_to_square(to_rgb(test[i]["image"])).convert("L").resize(
            cams[i].shape[::-1]))
        axes[0, k].imshow(img, cmap="gray")
        axes[1, k].imshow(img, cmap="gray")
        axes[1, k].imshow(cams[i], cmap="jet", alpha=0.45)
        axes[0, k].set_title(f"y={ys[i]} {'ok' if correct[i] else 'err'}", fontsize=7)
        axes[1, k].set_title(f"centre={masses[i]:.2f}", fontsize=7)
        for a in axes[:, k]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"gradcam_{tag}.png", dpi=150)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
