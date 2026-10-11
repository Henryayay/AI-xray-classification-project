"""Run bookkeeping: prediction files, provenance metadata and the test-set access log."""

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

PREDICTION_COLUMNS = ["split", "index", "y_true", "logit", "perturbation"]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_commit(repo_root=None):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment_info() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    for name in [
        "torch", "torchvision", "transformers", "datasets", "numpy", "scikit-learn",
        "unsloth", "trl", "peft", "bitsandbytes", "torchxrayvision",
    ]:
        try:
            from importlib.metadata import version

            info["packages"][name] = version(name)
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def run_dir(results_root, task, exp_name, run_name) -> Path:
    d = Path(results_root) / task / exp_name / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_predictions(path: Path, df: pd.DataFrame):
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"predictions missing columns {missing}")
    df.to_parquet(path / "predictions.parquet", index=False)


def write_meta(path: Path, meta: dict):
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def log_test_access(results_root, task, exp_name, run_name, cfg_hash):
    """Append-only record of every evaluation that touched the test split."""
    entry = {
        "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "task": task,
        "experiment": exp_name,
        "run": run_name,
        "config_hash": cfg_hash,
        "git_commit": git_commit(),
        "host": platform.node(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
    }
    p = Path(results_root) / "test_access_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_runs(results_root):
    """Load every ``predictions.parquet`` below ``results_root`` with its metadata."""
    runs = []
    for pred in sorted(Path(results_root).glob("*/*/*/predictions.parquet")):
        meta_p = pred.parent / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        task, exp, run = pred.parts[-4], pred.parts[-3], pred.parts[-2]
        runs.append({"task": task, "experiment": exp, "run": run, "meta": meta,
                     "predictions": pd.read_parquet(pred)})
    return runs
