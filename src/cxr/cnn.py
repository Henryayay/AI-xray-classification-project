"""Supervised CNN baselines: training with validation-based model selection."""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import pad_to_square, to_rgb
from .metrics import all_metrics
from .robustness import perturb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ARCHS = ("resnet18", "resnet50", "densenet121", "xrv_densenet121")


def build_model(arch: str, pretrained=True):
    from torchvision import models

    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained
                               else None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif arch == "xrv_densenet121":
        import torchxrayvision as xrv

        m = XRVWrapper(xrv.models.DenseNet(weights="densenet121-res224-all"))
    else:
        raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
    return m


class XRVWrapper(nn.Module):
    """TorchXRayVision DenseNet pretrained on large CXR corpora, with a new 2-way head.

    Expects single-channel inputs scaled to [-1024, 1024] as in the original work.
    """

    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.head = nn.Linear(base.classifier.in_features, 2)

    def forward(self, x):
        f = torch.relu(self.features(x))
        f = torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1)
        return self.head(f)


def last_conv_layer(model):
    return model.layer4 if hasattr(model, "layer4") else model.features


def _prep_rgb(im):
    return pad_to_square(to_rgb(im))


def _prep_gray(im):
    return pad_to_square(to_rgb(im).convert("L"))


def _xrv_scale(t):
    return (t * 2 - 1) * 1024


def make_transform(arch, size, train):
    from torchvision import transforms as T

    xrv = arch.startswith("xrv")
    ops = [T.Lambda(_prep_gray if xrv else _prep_rgb), T.Resize((size, size))]
    if train:
        ops += [T.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.93, 1.07)),
                T.ColorJitter(brightness=0.1, contrast=0.1)]
    ops.append(T.ToTensor())
    if xrv:
        ops.append(T.Lambda(_xrv_scale))
    else:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)


class HFImageDataset(Dataset):
    def __init__(self, split, indices, transform, perturbation="none"):
        self.split = split
        self.indices = np.asarray(indices)
        self.transform = transform
        self.perturbation = perturbation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        ex = self.split[idx]
        img = perturb(ex["image"], self.perturbation, seed=idx)
        return self.transform(img), int(ex["label"]), idx


def _seed_everything(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, logits, idxs = [], [], []
    for x, y, idx in loader:
        out = model(x.to(device, non_blocking=True)).float()
        logits.append((out[:, 1] - out[:, 0]).cpu().numpy())
        ys.append(y.numpy())
        idxs.append(idx.numpy())
    return np.concatenate(idxs), np.concatenate(ys), np.concatenate(logits)


def train_cnn(ds, cfg, seed, train_indices, device, log=print):
    """Train one model; return (best model, history, selection summary).

    The checkpoint with the lowest validation loss is kept; ties in loss are
    rare, and validation AUROC is logged for transparency only.
    """
    _seed_everything(seed)
    arch, size = cfg["arch"], cfg.get("image_size", 224)
    bs, epochs = cfg.get("batch_size", 32), cfg.get("epochs", 30)
    patience = cfg.get("patience", 7)
    workers = cfg.get("num_workers", 4)

    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(HFImageDataset(ds["train"], train_indices, make_transform(arch, size, True)),
                    batch_size=bs, shuffle=True, generator=g, num_workers=workers,
                    worker_init_fn=_worker_init, pin_memory=device.type == "cuda",
                    drop_last=len(train_indices) > bs)
    va = DataLoader(HFImageDataset(ds["validation"], np.arange(len(ds["validation"])),
                                   make_transform(arch, size, False)),
                    batch_size=bs * 2, shuffle=False, num_workers=workers)

    model = build_model(arch, pretrained=cfg.get("pretrained", True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",
                                                                                   1e-4))
    steps_per_epoch = max(1, len(tr))
    total = steps_per_epoch * epochs
    warm = steps_per_epoch * cfg.get("warmup_epochs", 1)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    use_amp = device.type == "cuda"

    best = {"val_loss": float("inf")}
    best_state, history, bad = None, [], 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        _, vy, vl = predict(model, va, device)
        val_loss = float(np.mean(np.logaddexp(0, vl) - vy * vl))
        vm = all_metrics(vy, vl)
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": val_loss,
               "val_auroc": vm["auroc"], "val_accuracy": vm["accuracy"],
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(rec)
        log(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4g}" for k, v in rec.items() if k != "epoch"))
        if val_loss < best["val_loss"] - 1e-4:
            best = dict(rec)
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, history, best


def eval_loader(split, arch, size, bs, perturbation="none", workers=4):
    return DataLoader(HFImageDataset(split, np.arange(len(split)), make_transform(arch, size, False),
                                     perturbation), batch_size=bs, shuffle=False,
                      num_workers=workers)


class GradCAM:
    """Minimal Grad-CAM (Selvaraju et al., 2017) for the positive-class logit margin."""

    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o):
        self.acts = o.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x):
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        (out[:, 1] - out[:, 0]).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))
        cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
        return cam.cpu().numpy(), (out[:, 1] - out[:, 0]).detach().cpu().numpy()
