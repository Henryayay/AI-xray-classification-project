"""Dataset audit: duplicate/near-duplicate leakage and low-level shortcut cues.

The POLCOVID splits published on the Hub carry no patient or hospital
identifiers, so patient-level leakage cannot be excluded from metadata. This
module provides the strongest checks available from pixels alone:

* exact duplicates (identical decoded pixels) within and across splits;
* near-duplicates by exhaustive pairwise Pearson correlation of normalised
  64x64 thumbnails, verified at 256x256. Perceptual hashes alone are not
  reliable here: chest radiographs share low-frequency structure, so 64-bit
  pHash distances <= 4 mostly flag different patients (verified empirically);
  pHash distance is therefore reported only as an attribute of each pair;
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
DUP_THUMB = 64
VERIFY_SIZE = 256
NEAR_IDENTICAL = 0.99
HIGHLY_SIMILAR = 0.95


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


def normalised_thumb(img, size):
    a = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() * np.sqrt(a.size) + 1e-6)


def compute_records(ds, phash_size=8):
    """One row per image across all splits with hashes and features.

    Returns (records, 8x8 probe thumbnails, normalised 64x64 duplicate thumbnails).
    """
    import imagehash

    rows, thumbs, dthumbs = [], [], []
    for split in ("train", "validation", "test"):
        for i, ex in enumerate(ds[split]):
            img = ex["image"]
            rec = {"split": split, "index": i, "label": int(ex["label"]),
                   "sha256": image_sha256(img),
                   "phash": str(imagehash.phash(img.convert("L"), hash_size=phash_size))}
            rec.update(image_features(img))
            rows.append(rec)
            thumbs.append(thumbnail(img))
            dthumbs.append(normalised_thumb(img, DUP_THUMB))
    return pd.DataFrame(rows), np.stack(thumbs), np.stack(dthumbs)


def _hamming_matrix(hex_hashes):
    bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
                     for h in hex_hashes], dtype=np.uint8)
    # Hamming distance via dot products of {0,1} vectors.
    b = bits.astype(np.int32)
    ones = b.sum(axis=1)
    return ones[:, None] + ones[None, :] - 2 * (b @ b.T)


def candidate_pairs(thumbs_a, thumbs_b=None, threshold=HIGHLY_SIMILAR - 0.03):
    """All (i, j) with thumbnail correlation above ``threshold`` (i < j when within one set)."""
    same = thumbs_b is None
    c = thumbs_a @ (thumbs_a if same else thumbs_b).T
    if same:
        c = np.triu(c, k=1)
    i, j = np.nonzero(c >= threshold)
    return i, j, c[i, j]


def verified_pairs(rec_a, ds_a, i, j, rec_b=None, ds_b=None, coarse=None):
    """Re-measure candidate pairs at ``VERIFY_SIZE`` and grade them."""
    rec_b = rec_a if rec_b is None else rec_b
    ds_b = ds_a if ds_b is None else ds_b
    ha = _hamming_matrix(rec_a.phash.tolist()) if rec_b is rec_a else None
    rows = []
    for k, (a, b) in enumerate(zip(i, j)):
        ra, rb = rec_a.iloc[a], rec_b.iloc[b]
        ia = ds_a[ra["split"]][int(ra["index"])]["image"]
        ib = ds_b[rb["split"]][int(rb["index"])]["image"]
        corr = float(normalised_thumb(ia, VERIFY_SIZE) @ normalised_thumb(ib, VERIFY_SIZE))
        if ha is not None:
            ham = int(ha[a, b])
        else:
            ham = int(_hamming_matrix([ra.phash, rb.phash])[0, 1])
        rows.append({
            "split_a": ra["split"], "index_a": int(ra["index"]), "label_a": int(ra["label"]),
            "split_b": rb["split"], "index_b": int(rb["index"]), "label_b": int(rb["label"]),
            "corr64": float(coarse[k]) if coarse is not None else np.nan, "corr256": corr,
            "phash_hamming": ham, "exact": bool(ra["sha256"] == rb["sha256"]),
        })
    df = pd.DataFrame(rows, columns=["split_a", "index_a", "label_a", "split_b", "index_b",
                                     "label_b", "corr64", "corr256", "phash_hamming", "exact"])
    df["grade"] = np.where(df.exact, "exact", np.where(
        df.corr256 >= NEAR_IDENTICAL, "near_identical",
        np.where(df.corr256 >= HIGHLY_SIMILAR, "highly_similar", "below_threshold")))
    return df[df.grade != "below_threshold"].sort_values("corr256", ascending=False)


def summarise_duplicates(pairs: pd.DataFrame) -> dict:
    out = {}
    for name, grades in (("near_identical_or_exact", ("exact", "near_identical")),
                         ("highly_similar_or_closer", ("exact", "near_identical",
                                                       "highly_similar"))):
        p = pairs[pairs.grade.isin(grades)]
        key = p.apply(lambda r: "-".join(sorted([r["split_a"], r["split_b"]])), axis=1) \
            if len(p) else pd.Series(dtype=str)
        cross = p[p.split_a != p.split_b]
        test_imgs = set()
        for _, r in cross.iterrows():
            for s_, i_, o in (("split_a", "index_a", "split_b"), ("split_b", "index_b", "split_a")):
                if r[s_] == "test" and r[o] == "train":
                    test_imgs.add(int(r[i_]))
        out[name] = {
            "n_pairs": int(len(p)),
            "by_split_pair": {k: int(v) for k, v in Counter(key).items()},
            "label_conflicts": int((p.label_a != p.label_b).sum()),
            "test_images_with_train_match": len(test_imgs),
        }
    return out


NUMERIC_PROBE_FEATURES = [
    "width", "height", "aspect", "log_pixels", "mean", "std", "p01", "p99", "frac_black",
    "frac_white", "border_mean", "border_std", "centre_mean",
] + [f"hist{i:02d}" for i in range(16)]


def phash_threshold_check(records: pd.DataFrame, thumbs64: np.ndarray, max_hamming=4) -> dict:
    """How often a small pHash distance corresponds to a genuinely near-identical image."""
    d = _hamming_matrix(records.phash.tolist())
    i, j = np.nonzero(np.triu(d <= max_hamming, k=1))
    corr = np.einsum("ij,ij->i", thumbs64[i], thumbs64[j])
    lab = records.label.to_numpy()
    return {"max_hamming": max_hamming, "n_pairs": int(len(i)),
            "n_corr64_ge_0.99": int(np.sum(corr >= NEAR_IDENTICAL)),
            "median_corr64": float(np.median(corr)) if len(corr) else None,
            "label_conflicts": int(np.sum(lab[i] != lab[j]))}


def shortcut_probe(records: pd.DataFrame, thumbs: np.ndarray, seed=0) -> dict:
    """Train simple classifiers on non-anatomical features; report val/test AUROC."""
    import warnings

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
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
