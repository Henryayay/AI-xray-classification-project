"""Deterministic test-time perturbations for robustness and shortcut analysis.

* ``border_mask``: blacken the outer frame, removing corner markers, text and
  collimation edges that lie outside the thorax.
* ``centre_mask``: blacken the central region containing most of the lung
  fields. A model that stays well above chance without the lungs is relying on
  non-anatomical cues (cf. DeGrave et al., 2021).
* ``gamma_low`` / ``gamma_high``: global contrast changes typical of different
  acquisition/windowing.
* ``noise``: additive Gaussian noise (fixed seed per image).
* ``jpeg``: strong JPEG re-compression.
"""

import io

import numpy as np
from PIL import Image

PERTURBATIONS = ("none", "border_mask", "centre_mask", "gamma_low", "gamma_high", "noise", "jpeg")


def perturb(img: Image.Image, kind: str, seed: int = 0) -> Image.Image:
    if kind == "none":
        return img
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if kind == "border_mask":
        fh, fw = int(0.125 * h), int(0.125 * w)
        out = np.zeros_like(g)
        out[fh:h - fh, fw:w - fw] = g[fh:h - fh, fw:w - fw]
    elif kind == "centre_mask":
        out = g.copy()
        out[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)] = 0
    elif kind == "gamma_low":
        out = 255.0 * (g / 255.0) ** 0.6
    elif kind == "gamma_high":
        out = 255.0 * (g / 255.0) ** 1.6
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        out = g + rng.normal(0, 0.05 * 255, size=g.shape)
    elif kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(g.astype(np.uint8)).save(buf, format="JPEG", quality=10)
        return Image.open(io.BytesIO(buf.getvalue())).convert(img.mode if img.mode in ("L", "RGB")
                                                               else "L")
    else:
        raise ValueError(f"unknown perturbation {kind}")
    out = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return out.convert("RGB") if img.mode == "RGB" else out
