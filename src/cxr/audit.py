"""Dataset audit: duplicate/near-duplicate leakage and low-level shortcut cues.

The POLCOVID splits published on the Hub carry no patient or hospital
identifiers, so patient-level leakage cannot be excluded from metadata. This
module provides the strongest checks available from pixels alone:

* exact duplicates (identical decoded pixels) within and across splits;
* near-duplicates by perceptual hash (pHash) Hamming distance, which catches
  re-encoded, resized or slightly re-windowed copies of the same radiograph;
* per-class distributions of image size, aspect ratio, bit depth/mode and
  intensity statistics; and
* "shortcut probes": simple classifiers trained on non-anatomical features
  (image geometry, intensity statistics, border statistics, tiny thumbnails).
  Above-chance validation/test AUROC from such features signals acquisition
  or site confounding that a deep model could exploit.
"""

from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from .data import image_sha256

BORDER_FRAC = 0.08
THUMB = 8


def image_features(img: Image.Image) -> dict:
    """Non-anatomical per-image descriptors used for the shortcut probe."""
    mode = img.mode
    w, h = img.size
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    bh, bw = max(1, int(h * BORDER_FRAC)), max(1, int(w * BORDER_FRAC))
    border = np.concatenate([g[:bh].ravel(), g[-bh:].ravel(), g[:, :bw].ravel(),
                             g[:, -bw:].ravel()])
    centre = g[bh:-bh, bw:-bw] if h > 2 * bh and w > 2 * bw else g
    hist = np.histogram(g, bins=16, range=(0, 1))[0] / g.size
    return {
        "width": w,
        "height": h,
        "aspect": w / h,
        "log_pixels": float(np.log(w * h)),
        "mode": mode,
        "mean": float(g.mean()),
        "std": float(g.std()),
        "p01": float(np.percentile(g, 1)),
        "p99": float(np.percentile(g, 99)),
        "frac_black": float(np.mean(g < 0.02)),
        "frac_white": float(np.mean(g > 0.98)),
        "border_mean": float(border.mean()),
        "border_std": float(border.std()),
        "centre_mean": float(centre.mean()),
        **{f"hist{i:02d}": float(v) for i, v in enumerate(hist)},
    }


def thumbnail(img: Image.Image, size=THUMB) -> np.ndarray:
    return np.asarray(img.convert("L").resize((size, size), Image.BILINEAR),
                      dtype=np.float32).ravel() / 255.0


def compute_records(ds, phash_size=8):
    """One row per image across all splits with hashes and features."""
    import imagehash

    rows, thumbs = [], []
    for split in ("train", "validation", "test"):
        for i, ex in enumerate(ds[split]):
            img = ex["image"]
            rec = {"split": split, "index": i, "label": int(ex["label"]),
                   "sha256": image_sha256(img),
                   "phash": str(imagehash.phash(img.convert("L"), hash_size=phash_size))}
            rec.update(image_features(img))
            rows.append(rec)
            thumbs.append(thumbnail(img))
    return pd.DataFrame(rows), np.stack(thumbs)


def _hamming_matrix(hex_hashes):
    bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
                     for h in hex_hashes], dtype=np.uint8)
    # Hamming distance via dot products of {0,1} vectors.
    b = bits.astype(np.int32)
    ones = b.sum(axis=1)
    return ones[:, None] + ones[None, :] - 2 * (b @ b.T)


def duplicate_pairs(records: pd.DataFrame, max_hamming=4):
    """Pairs of images that are exact or perceptual near-duplicates."""
    d = _hamming_matrix(records["phash"].tolist())
    iu = np.triu_indices(len(records), k=1)
    close = d[iu] <= max_hamming
    a, b = iu[0][close], iu[1][close]
    out = []
    sha = records["sha256"].to_numpy()
    for i, j in zip(a, b):
        ri, rj = records.iloc[i], records.iloc[j]
        out.append({
            "split_a": ri["split"], "index_a": int(ri["index"]), "label_a": int(ri["label"]),
            "split_b": rj["split"], "index_b": int(rj["index"]), "label_b": int(rj["label"]),
            "hamming": int(d[i, j]), "exact": bool(sha[i] == sha[j]),
        })
    # Exact duplicates whose pHash differs by more than max_hamming cannot exist,
    # since identical pixels give identical hashes.
    return pd.DataFrame(out, columns=["split_a", "index_a", "label_a", "split_b", "index_b",
                                      "label_b", "hamming", "exact"])


def summarise_duplicates(pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        return {"n_pairs": 0}
    key = pairs.apply(lambda r: "-".join(sorted([r["split_a"], r["split_b"]])), axis=1)
    return {
        "n_pairs": int(len(pairs)),
        "n_exact_pairs": int(pairs["exact"].sum()),
        "by_split_pair": dict(Counter(key)),
        "label_conflicts": int((pairs["label_a"] != pairs["label_b"]).sum()),
        "test_images_with_train_neighbour": int(len(set(
            pairs.loc[(pairs.split_a == "test") & (pairs.split_b == "train"), "index_a"]) | set(
            pairs.loc[(pairs.split_b == "test") & (pairs.split_a == "train"), "index_b"]))),
    }


NUMERIC_PROBE_FEATURES = [
    "width", "height", "aspect", "log_pixels", "mean", "std", "p01", "p99", "frac_black",
    "frac_white", "border_mean", "border_std", "centre_mean",
] + [f"hist{i:02d}" for i in range(16)]


def shortcut_probe(records: pd.DataFrame, thumbs: np.ndarray, seed=0) -> dict:
    """Train simple classifiers on non-anatomical features; report val/test AUROC."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = {
        "geometry": ["width", "height", "aspect", "log_pixels"],
        "intensity_stats": NUMERIC_PROBE_FEATURES[4:],
        "border_only": ["border_mean", "border_std", "frac_black", "frac_white"],
        "all_tabular": NUMERIC_PROBE_FEATURES,
    }
    mode_dummies = pd.get_dummies(records["mode"], prefix="mode").astype(float)
    tr = (records["split"] == "train").to_numpy()
    out = {}
    for name, cols in list(feats.items()) + [("mode", None), ("thumbnail_8x8", None)]:
        if name == "mode":
            X = mode_dummies.to_numpy()
        elif name == "thumbnail_8x8":
            X = thumbs
        else:
            X = records[cols].to_numpy(dtype=float)
        y = records["label"].to_numpy()
        res = {}
        models = {
            "logreg": make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=2000, cv=5)),
            "gbm": GradientBoostingClassifier(random_state=seed),
        }
        for mname, model in models.items():
            if X.shape[1] == 0 or len(np.unique(X[tr], axis=0)) < 2:
                res[mname] = {"validation_auroc": 0.5, "test_auroc": 0.5}
                continue
            model.fit(X[tr], y[tr])
            res[mname] = {}
            for split in ("validation", "test"):
                m = (records["split"] == split).to_numpy()
                s = model.predict_proba(X[m])[:, 1]
                res[mname][f"{split}_auroc"] = float(roc_auc_score(y[m], s))
        out[name] = res
    return out


def class_summary(records: pd.DataFrame) -> pd.DataFrame:
    cols = ["width", "height", "aspect", "mean", "std", "border_mean", "frac_black"]
    g = records.groupby(["split", "label"])
    summary = g[cols].median()
    summary["n"] = g.size()
    modes = records.groupby(["split", "label"])["mode"].agg(lambda s: dict(Counter(s)))
    summary["modes"] = modes
    return summary.reset_index()
