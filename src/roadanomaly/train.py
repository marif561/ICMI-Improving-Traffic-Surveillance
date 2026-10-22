"""Fine-tune a pre-trained CNN for road anomaly classification.

Transfer learning from ImageNet weights, data augmentation (random crop/zoom, flips,
rotation, shift, shear, colour jitter), SGD with momentum and step learning-rate decay.
The checkpoint with the best validation accuracy is kept and evaluated on the test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .data import ImageDataset, build_transforms, make_splits
from .models import build_model, default_image_size, main_logits


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(main_logits(model(x.to(device))), -1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def report(y, probs, classes):
    pred = probs.argmax(-1)
    ids = list(range(len(classes)))
    return {"accuracy": accuracy_score(y, pred),
            "per_class": classification_report(y, pred, labels=ids, target_names=classes, output_dict=True,
                                               zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=ids).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="image-folder root (class subfolders, or train/val/test)")
    ap.add_argument("--backbone", default=None, help="inception_v3 | resnet50 | vgg16 | mobilenet_v2 | densenet121")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (for tests)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    name = args.backbone or cfg["backbone"]
    size = args.image_size or cfg.get("image_size") or default_image_size(name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits, classes = make_splits(args.data, cfg["val_size"], cfg["test_size"], args.seed)
    print({k: len(v) for k, v in splits.items()}, "classes:", classes)
    tf_train = build_transforms(size, cfg["augmentation"], train=True)
    tf_eval = build_transforms(size, cfg["augmentation"], train=False)
    loader = lambda items, tf, shuffle: DataLoader(ImageDataset(items, tf), batch_size=cfg["batch_size"],  # noqa: E731
                                                   shuffle=shuffle, drop_last=shuffle and len(items) > cfg["batch_size"],
                                                   num_workers=args.workers)
    train_dl = loader(splits["train"], tf_train, True)
    eval_dl = {s: loader(splits[s], tf_eval, False) for s in ("val", "test") if s in splits}

    pretrained = cfg.get("pretrained", True) and not args.no_pretrained
    model = build_model(name, len(classes), pretrained, cfg["dropout"], cfg["freeze_backbone"]).to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(cfg["learning_rate"]),
                          momentum=cfg["momentum"], weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    for epoch in range(1, (args.epochs or cfg["epochs"]) + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            outp = model(x)
            loss = loss_fn(main_logits(outp), y)
            if hasattr(outp, "aux_logits") and outp.aux_logits is not None:
                loss = loss + 0.4 * loss_fn(outp.aux_logits, y)  # InceptionV3 auxiliary head
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(y), n + len(y)
        sched.step()
        msg = f"epoch {epoch:>3}  loss {total / max(n, 1):.4f}"
        val_acc = report(*predict(model, eval_dl["val"], device), classes)["accuracy"] if "val" in eval_dl else 0.0
        msg += f"  val acc {val_acc:.4f}" if "val" in eval_dl else ""
        print(msg)
        if val_acc >= best:
            best = val_acc
            torch.save(model.state_dict(), out / "model.pt")

    (out / "meta.json").write_text(json.dumps({"backbone": name, "classes": classes, "image_size": size,
                                               "dropout": cfg["dropout"]}, indent=2))
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    for split, dl in eval_dl.items():
        res = report(*predict(model, dl, device), classes)
        print(f"{split} accuracy {res['accuracy']:.4f}")
        (out / f"{split}_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
