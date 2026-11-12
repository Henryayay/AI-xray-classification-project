#!/usr/bin/env python
"""Run every condition of one experiment config and write predictions + provenance.

    python scripts/run_experiment.py --config configs/cnn_resnet50.yaml            # validation only
    python scripts/run_experiment.py --config configs/cnn_resnet50.yaml --final    # also test split

Without ``--final`` only the validation split is scored, so model and
hyperparameter choices can be made without touching the test set. ``--final``
additionally scores the test split (and any configured perturbations) and
appends an entry to ``results/test_access_log.jsonl``.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr import io as rio  # noqa: E402
from cxr.data import dataset_revision, labels_of, load_splits, stratified_subset  # noqa: E402
from cxr.tasks import get_task  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def limit_split(split, n):
    return split if not n or n >= len(split) else split.select(range(n))


def eval_splits(final):
    return ("validation", "test") if final else ("validation",)


def frame(split_name, idx, y, logit, perturbation="none", **extra):
    d = {"split": split_name, "index": np.asarray(idx, int), "y_true": np.asarray(y, int),
         "logit": np.asarray(logit, float), "perturbation": perturbation}
    d.update({k: np.asarray(v) for k, v in extra.items()})
    return pd.DataFrame(d)


# --------------------------------------------------------------------------- CNN

def select_lr(ds, cfg, task, exp_dir, device, train_idx):
    if cfg.get("lr_from"):
        src = exp_dir.parent / cfg["lr_from"] / "hparam_selection.json"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found; run {cfg['lr_from']} first")
        sel = json.loads(src.read_text())
        return sel["lr"], {**sel, "inherited_from": cfg["lr_from"]}
    grid = cfg.get("lr_grid") or [cfg["lr"]]
    sel_p = exp_dir / "hparam_selection.json"
    if len(grid) == 1:
        return grid[0], None
    if sel_p.exists():
        return json.loads(sel_p.read_text())["lr"], json.loads(sel_p.read_text())
    from cxr.cnn import train_cnn

    results = []
    for lr in grid:
        log(f"[lr sweep] lr={lr}")
        _, hist, best = train_cnn(ds, {**cfg, "lr": lr}, cfg.get("sweep_seed", 1000), train_idx,
                                  device, log)
        results.append({"lr": lr, "best_val_loss": best["val_loss"],
                        "best_val_auroc": best["val_auroc"], "best_epoch": best["epoch"]})
    choice = min(results, key=lambda r: r["best_val_loss"])
    sel = {"lr": choice["lr"], "criterion": "min validation loss", "sweep_seed":
           cfg.get("sweep_seed", 1000), "grid": results}
    sel_p.write_text(json.dumps(sel, indent=2))
    return choice["lr"], sel


def run_cnn(ds, task, cfg, seed, fraction, out_dir, exp_dir, final, device):
    import torch

    from cxr.cnn import build_model, eval_loader, predict, train_cnn

    train_idx = stratified_subset(labels_of(ds["train"]), fraction, seed)
    lr, sel = select_lr(ds, cfg, task, exp_dir, device, stratified_subset(
        labels_of(ds["train"]), fraction, cfg.get("sweep_seed", 1000)))
    ckpt = out_dir / "model.pt"
    meta = {"train_size": int(len(train_idx)), "lr": lr, "hparam_selection": sel}
    if ckpt.exists():
        model = build_model(cfg["arch"], pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        meta.update(json.loads((out_dir / "train_log.json").read_text()))
    else:
        model, hist, best = train_cnn(ds, {**cfg, "lr": lr}, seed, train_idx, device, log)
        torch.save(model.state_dict(), ckpt)
        tl = {"history": hist, "selected_epoch": best}
        (out_dir / "train_log.json").write_text(json.dumps(tl, indent=2))
        meta.update(tl)
    frames = []
    for s in eval_splits(final):
        perts = ["none"] + (cfg.get("perturbations", []) if s == "test" else [])
        for p in perts:
            idx, y, lg = predict(model, eval_loader(ds[s], cfg["arch"], cfg.get("image_size", 224),
                                                    cfg.get("batch_size", 32) * 2, p,
                                                    cfg.get("num_workers", 4)), device)
            frames.append(frame(s, idx, y, lg, p))
    return pd.concat(frames), meta


# ------------------------------------------------------------------ linear probe

def run_probe(ds, task, cfg, seed, fraction, out_dir, exp_dir, final, device, cache_dir):
    from cxr.embed import cached_embeddings, fit_linear_probe, probe_logits

    emb = cached_embeddings(cache_dir, task.name, cfg["encoder"], ds, device,
                            splits=("train",) + eval_splits(final))
    ytr = labels_of(ds["train"])
    tr = stratified_subset(ytr, fraction, seed)
    yva = labels_of(ds["validation"])
    probe = fit_linear_probe(emb["train"][tr], ytr[tr], emb["validation"], yva, seed=seed)
    frames = []
    for s in eval_splits(final):
        y = labels_of(ds[s])
        frames.append(frame(s, np.arange(len(y)), y, probe_logits(probe, emb[s])))
        if s == "test":
            for p in cfg.get("perturbations", []):
                e = cached_embeddings(cache_dir, task.name, cfg["encoder"], ds, device,
                                      splits=("test",), perturbation=p)["test"]
                frames.append(frame(s, np.arange(len(y)), y, probe_logits(probe, e), p))
    return pd.concat(frames), {"train_size": int(len(tr)), "C": probe["C"],
                               "val_nll": probe["val_nll"]}


# ------------------------------------------------------------------------- VLMs

def _score_split(scorer, task, split, s_name, messages_fn, cfg, perturbation="none",
                 content_free=None):
    """Score one split; ``messages_fn(i, query_image) -> (messages, images)``."""
    from tqdm import tqdm

    from cxr.robustness import perturb
    from cxr.vlm import blank_like, parse_label

    rows = {"index": [], "y": [], "lp0": [], "lp1": [], "cf": [], "gen": [], "parsed": []}
    for i in tqdm(range(len(split)), desc=f"{s_name}/{perturbation}", mininterval=30):
        ex = split[i]
        img = perturb(ex["image"], perturbation, seed=i)
        msgs, imgs = messages_fn(i, img)
        lp = scorer.score(msgs, imgs)
        rows["index"].append(i)
        rows["y"].append(int(ex["label"]))
        rows["lp0"].append(lp[0])
        rows["lp1"].append(lp[1])
        if content_free == "per_query":
            m2, i2 = messages_fn(i, blank_like(img))
            c = scorer.score(m2, i2)
            rows["cf"].append(c[1] - c[0])
        else:
            rows["cf"].append(np.nan)
        if cfg.get("also_generate") and perturbation == "none":
            g = scorer.generate(msgs, imgs)
            rows["gen"].append(g)
            rows["parsed"].append(parse_label(g, task.class_names))
        else:
            rows["gen"].append("")
            rows["parsed"].append(-100)
    lp0, lp1 = np.array(rows["lp0"]), np.array(rows["lp1"])
    return frame(s_name, rows["index"], rows["y"], lp1 - lp0, perturbation, logp_neg=lp0,
                 logp_pos=lp1, logit_cf=np.array(rows["cf"], float), generated=rows["gen"],
                 parsed_label=rows["parsed"])


def _content_free_constant(scorer, messages_fn, ref_image):
    from cxr.vlm import blank_like

    m, i = messages_fn(-1, blank_like(ref_image))
    c = scorer.score(m, i)
    return float(c[1] - c[0])


def run_vlm(ds, task, cfg, seed, fraction, out_dir, exp_dir, final, device, cache_dir,
            shots=None, scorer_cache={}):
    import torch

    from cxr import prompts
    from cxr.icl import ExemplarSelector
    from cxr.vlm import LabelScorer, load_vlm, train_lora

    kind = cfg["kind"]
    meta = {}
    adapter = None
    if kind == "vlm_lora":
        tr = stratified_subset(labels_of(ds["train"]), fraction, seed)
        adapter = out_dir / "adapter"
        meta["train_size"] = int(len(tr))
        if not adapter.exists():
            adapter, hist, best = train_lora(task, ds["train"], ds["validation"], cfg, seed,
                                             out_dir, train_indices=tr, log=log)
            (out_dir / "train_log.json").write_text(json.dumps(
                {"history": hist, **best}, indent=2, default=str))
        meta.update(json.loads((out_dir / "train_log.json").read_text()))
        scorer_cache.clear()
        torch.cuda.empty_cache()

    key = (cfg["model_id"], str(adapter))
    if key not in scorer_cache:
        scorer_cache.clear()
        torch.cuda.empty_cache()
        model, proc = load_vlm(cfg["model_id"], device=device, adapter_dir=adapter,
                               revision=cfg.get("model_revision"))
        scorer_cache[key] = LabelScorer(model, proc, task.class_names,
                                        use_prefix_cache=cfg.get("prefix_cache", True))
    scorer = scorer_cache[key]
    train = ds["train"]
    cf_mode = cfg.get("content_free", "per_draw")

    if kind == "vlm_icl" and shots:
        strategy = cfg["strategy"]
        emb = None
        if strategy == "knn":
            from cxr.embed import cached_embeddings

            emb = cached_embeddings(cache_dir, task.name, cfg["encoder"], ds, device,
                                    splits=("train",) + eval_splits(final))
        selector = ExemplarSelector(strategy, labels_of(train), shots, seed,
                                    train_emb=None if emb is None else emb["train"],
                                    max_similarity=cfg.get("max_similarity"))
        names = task.class_names
        meta["shots"] = shots
        if strategy == "random":
            meta["exemplar_train_indices"] = selector.fixed.tolist()

        def make_fn(split_name):
            def fn(i, img):
                q = None if emb is None or i < 0 else emb[split_name][i]
                ex_idx = selector.select(q, query_key=i)
                labels = [names[int(train[int(j)]["label"])] for j in ex_idx]
                imgs = [train[int(j)]["image"] for j in ex_idx] + [img]
                return prompts.icl_messages(task, labels), imgs
            return fn
    else:
        def make_fn(split_name):
            return lambda i, img: (prompts.zero_shot_messages(task), [img])

    frames = []
    for s in eval_splits(final):
        fn = make_fn(s)
        per_query = cf_mode == "per_query"
        f = _score_split(scorer, task, ds[s], s, fn, cfg, content_free=cf_mode)
        if cf_mode == "per_draw":
            f["logit_cf"] = _content_free_constant(scorer, fn, ds[s][0]["image"])
        frames.append(f)
        if s == "test":
            for p in cfg.get("perturbations", []):
                fp = _score_split(scorer, task, ds[s], s, fn, cfg, perturbation=p,
                                  content_free="per_query" if per_query else None)
                if cf_mode == "per_draw":
                    fp["logit_cf"] = frames[-1]["logit_cf"].iloc[0]
                frames.append(fp)
    return pd.concat(frames), meta


# ------------------------------------------------------------------------- main

def expand(cfg):
    seeds = cfg.get("seeds", [0])
    fracs = cfg.get("train_fractions", [1.0])
    shots = cfg.get("shots", [None])
    for seed, frac, n in itertools.product(seeds, fracs, shots):
        name = f"seed{seed}"
        if frac != 1.0:
            name += f"_frac{frac:g}"
        if n is not None:
            name += f"_shots{n}"
        yield seed, frac, n, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--final", action="store_true", help="also score the test split")
    ap.add_argument("--tasks", nargs="*", help="override tasks in config")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per split")
    ap.add_argument("--only-runs", nargs="*", help="restrict to these run names")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["name"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    chash = rio.config_hash(cfg)
    env = rio.environment_info()
    results_root = Path(args.results)
    if args.limit:
        results_root = results_root / "_smoke"

    for task_name in args.tasks or cfg["tasks"]:
        task = get_task(task_name)
        ds = load_splits(task)
        if args.limit:
            ds = {k: limit_split(v, args.limit) for k, v in ds.items()}
        rev = dataset_revision(task)
        exp_dir = results_root / task.name / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        for seed, frac, shots, run_name in expand(cfg):
            if args.only_runs and run_name not in args.only_runs:
                continue
            out = rio.run_dir(results_root, task.name, exp, run_name)
            mp = out / "meta.json"
            if mp.exists() and not args.overwrite:
                prev = json.loads(mp.read_text())
                if prev.get("final") or not args.final:
                    log(f"skip {task.name}/{exp}/{run_name} (exists)")
                    continue
            log(f"run {task.name}/{exp}/{run_name} final={args.final}")
            t0 = time.time()
            kind = cfg["kind"]
            if kind == "cnn":
                preds, meta = run_cnn(ds, task, cfg, seed, frac, out, exp_dir, args.final, device)
            elif kind == "linear_probe":
                preds, meta = run_probe(ds, task, cfg, seed, frac, out, exp_dir, args.final,
                                        device, args.cache)
            elif kind in ("vlm_zeroshot", "vlm_icl", "vlm_lora"):
                preds, meta = run_vlm(ds, task, cfg, seed, frac, out, exp_dir, args.final, device,
                                      args.cache, shots=shots)
            else:
                raise ValueError(f"unknown kind {kind}")
            rio.write_predictions(out, preds)
            rio.write_meta(out, {
                "task": task.name, "experiment": exp, "run": run_name, "kind": kind,
                "seed": seed, "train_fraction": frac, "shots": shots, "final": args.final,
                "limit": args.limit, "config": cfg, "config_hash": chash,
                "config_path": str(Path(args.config)), "git_commit": rio.git_commit(ROOT),
                "dataset_id": task.dataset_id, "dataset_revision": rev, "environment": env,
                "wall_seconds": time.time() - t0, **meta,
            })
            if args.final:
                rio.log_test_access(results_root, task.name, exp, run_name, chash)
            log(f"done {run_name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
