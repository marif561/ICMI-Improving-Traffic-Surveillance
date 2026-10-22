"""Image-folder data: listing, stratified splits and augmentation.

Layout: ``<root>/<class_name>/*.jpg``. If ``<root>`` contains ``train/`` (and optionally
``val/`` and ``test/``), those splits are used as given; otherwise a stratified split is made.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
Items = List[Tuple[str, int]]


def list_images(root: str | Path, classes: List[str] | None = None) -> Tuple[Items, List[str]]:
    root = Path(root)
    classes = classes or sorted(p.name for p in root.iterdir() if p.is_dir())
    items = [(str(f), i) for i, c in enumerate(classes) for f in sorted((root / c).rglob("*"))
             if f.suffix.lower() in IMG_EXT]
    if not items:
        raise FileNotFoundError(f"No images found under {root}")
    return items, classes


def make_splits(root: str | Path, val_size: float = 0.15, test_size: float = 0.15,
                seed: int = 42) -> Tuple[Dict[str, Items], List[str]]:
    root = Path(root)
    if (root / "train").is_dir():
        train, classes = list_images(root / "train")
        splits = {"train": train}
        for s in ("val", "test"):
            if (root / s).is_dir():
                splits[s] = list_images(root / s, classes)[0]
        return splits, classes
    items, classes = list_images(root)
    y = [l for _, l in items]
    rest, test = train_test_split(items, test_size=test_size, random_state=seed, stratify=y)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(rest, test_size=rel_val, random_state=seed, stratify=[l for _, l in rest])
    return {"train": train, "val": val, "test": test}, classes


def build_transforms(image_size: int, aug: dict, train: bool):
    from torchvision import transforms as T

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet statistics
    if not train:
        return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), norm])
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(1 - aug.get("zoom", 0.15), 1.0)),
        T.RandomHorizontalFlip(p=0.5 if aug.get("hflip", True) else 0.0),
        T.RandomAffine(degrees=aug.get("rotation", 30), translate=(aug.get("shift", 0.2),) * 2,
                       shear=aug.get("shear", 15)),
        T.ColorJitter(*(aug.get("color_jitter", [0.2, 0.2, 0.2]))),
        T.ToTensor(), norm])


class ImageDataset:
    """Minimal torch-compatible dataset over (path, label) items."""

    def __init__(self, items: Items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label
