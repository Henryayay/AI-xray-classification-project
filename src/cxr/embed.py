"""Frozen image encoders: embeddings for linear probes and ICL exemplar retrieval."""

from pathlib import Path

import numpy as np
import torch

from .data import pad_to_square, to_rgb

ENCODERS = {
    "rad_dino": "microsoft/rad-dino",
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "imagenet_resnet50": None,
}


class Encoder:
    def __init__(self, name, device):
        self.name = name
        self.device = device
        if name == "rad_dino":
            from transformers import AutoImageProcessor, AutoModel

            self.proc = AutoImageProcessor.from_pretrained(ENCODERS[name])
            self.model = AutoModel.from_pretrained(ENCODERS[name]).to(device).eval()
        elif name == "biomedclip":
            import open_clip

            self.model, self.proc = open_clip.create_model_from_pretrained(ENCODERS[name])
            self.model = self.model.to(device).eval()
        elif name == "imagenet_resnet50":
            from torchvision import models

            w = models.ResNet50_Weights.IMAGENET1K_V2
            m = models.resnet50(weights=w)
            m.fc = torch.nn.Identity()
            self.model = m.to(device).eval()
            self.proc = w.transforms()
        else:
            raise ValueError(f"unknown encoder {name}; choose from {sorted(ENCODERS)}")

    @torch.no_grad()
    def __call__(self, images):
        images = [pad_to_square(to_rgb(im)) for im in images]
        if self.name == "rad_dino":
            x = self.proc(images=images, return_tensors="pt").to(self.device)
            out = self.model(**x)
            f = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0]
        elif self.name == "biomedclip":
            x = torch.stack([self.proc(im) for im in images]).to(self.device)
            f = self.model.encode_image(x)
        else:
            x = torch.stack([self.proc(im) for im in images]).to(self.device)
            f = self.model(x)
        return f.float().cpu().numpy()


def embed_split(encoder, split, batch_size=32, perturbation="none"):
    from .robustness import perturb

    feats = []
    for start in range(0, len(split), batch_size):
        batch = split[start:start + batch_size]
        imgs = [perturb(im, perturbation, seed=start + j) for j, im in enumerate(batch["image"])]
        feats.append(encoder(imgs))
    return np.concatenate(feats)


def cached_embeddings(cache_dir, task_name, encoder_name, ds, device, splits=None,
                      perturbation="none"):
    """Embeddings per split, cached as ``.npy`` so retrieval and probes reuse them."""
    splits = splits or ("train", "validation", "test")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, enc = {}, None
    for s in splits:
        suffix = "" if perturbation == "none" else f"_{perturbation}"
        p = cache_dir / f"{task_name}_{encoder_name}_{s}{suffix}.npy"
        if p.exists():
            out[s] = np.load(p)
            continue
        enc = enc or Encoder(encoder_name, device)
        out[s] = embed_split(enc, ds[s], perturbation=perturbation)
        np.save(p, out[s])
    return out


def l2_normalise(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def fit_linear_probe(train_x, train_y, val_x, val_y, seed=0, Cs=None):
    """L2 logistic regression; inverse regularisation strength chosen on validation NLL."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Cs = Cs or [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    scaler = StandardScaler().fit(train_x)
    tx, vx = scaler.transform(train_x), scaler.transform(val_x)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(tx, train_y)
        z = clf.decision_function(vx)
        nll = float(np.mean(np.logaddexp(0, z) - val_y * z))
        if best is None or nll < best[0]:
            best = (nll, C, clf)
    _, C, clf = best
    return {"scaler": scaler, "clf": clf, "C": C, "val_nll": best[0]}


def probe_logits(probe, x):
    return probe["clf"].decision_function(probe["scaler"].transform(x))
