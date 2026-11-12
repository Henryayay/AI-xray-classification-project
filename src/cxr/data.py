"""Dataset loading, deterministic subsampling and image preprocessing."""

import hashlib
import io

import numpy as np
from PIL import Image, ImageOps

from .tasks import Task

SPLITS = ("train", "validation", "test")


def load_splits(task: Task, revision=None, cache_dir=None):
    """Return a ``DatasetDict`` with train/validation/test splits for ``task``."""
    from datasets import load_dataset

    ds = load_dataset(task.dataset_id, revision=revision or task.revision, cache_dir=cache_dir)
    missing = [s for s in SPLITS if s not in ds]
    if missing:
        raise ValueError(f"{task.dataset_id} is missing splits {missing}")
    return ds


def dataset_revision(task: Task):
    """Pinned Hub commit used for every experiment (see ``tasks.py``)."""
    return task.revision


def labels_of(split) -> np.ndarray:
    return np.asarray(split["label"], dtype=np.int64)


def image_sha256(img: Image.Image) -> str:
    """Hash of decoded pixel content, independent of the container format."""
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    h = hashlib.sha256()
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def stratified_subset(labels, fraction: float, seed: int) -> np.ndarray:
    """Indices of a class-stratified random subset containing ``fraction`` of each class."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    labels = np.asarray(labels)
    if fraction == 1:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        n = max(1, int(round(fraction * len(idx))))
        keep.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(keep))


def pad_to_square(img: Image.Image, fill=0) -> Image.Image:
    """Letterbox to a square canvas so resizing does not distort anatomy."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    return ImageOps.pad(img, (side, side), color=fill, centering=(0.5, 0.5))


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("I;16", "I"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, [0.5, 99.5])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGB")


def image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))
