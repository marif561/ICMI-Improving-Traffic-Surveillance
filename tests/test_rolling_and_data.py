import numpy as np
import pytest
from PIL import Image

from roadanomaly.data import list_images, make_splits
from roadanomaly.rolling import RollingPredictor, flicker_rate, smooth_sequence

CLASSES = ["accident", "car_fire", "fighting", "snatching"]


def test_rolling_average_and_window():
    rp = RollingPredictor(window=2)
    assert np.allclose(rp.update([1, 0]), [1, 0])
    assert np.allclose(rp.update([0, 1]), [0.5, 0.5])
    assert np.allclose(rp.update([0, 1]), [0, 1])  # first frame has left the window
    with pytest.raises(ValueError):
        RollingPredictor(0)


def test_rolling_reduces_flicker():
    # true class 0 throughout, but single-frame predictions flip to class 1 now and then
    probs = np.tile([0.7, 0.3], (60, 1))
    probs[[5, 12, 13, 20, 27, 33, 41, 42, 50, 57]] = [0.3, 0.7]
    raw = probs.argmax(1).tolist()
    smooth = smooth_sequence(probs, window=8)
    assert flicker_rate(smooth) < flicker_rate(raw)
    assert set(smooth) == {0}


def _make_images(root, per_class=8):
    for c_i, c in enumerate(CLASSES):
        (root / c).mkdir(parents=True)
        for i in range(per_class):
            arr = np.full((32, 32, 3), 40 * c_i + i, dtype=np.uint8)
            Image.fromarray(arr).save(root / c / f"{i}.png")


def test_image_listing_and_stratified_splits(tmp_path):
    _make_images(tmp_path)
    items, classes = list_images(tmp_path)
    assert classes == CLASSES and len(items) == 32
    splits, _ = make_splits(tmp_path, val_size=0.25, test_size=0.25)
    assert sum(len(v) for v in splits.values()) == 32
    for s in ("train", "val", "test"):
        assert {l for _, l in splits[s]} == {0, 1, 2, 3}  # every class in every split


def test_predefined_splits(tmp_path):
    _make_images(tmp_path / "train", 4)
    _make_images(tmp_path / "test", 2)
    splits, classes = make_splits(tmp_path)
    assert set(splits) == {"train", "test"} and len(splits["test"]) == 8
