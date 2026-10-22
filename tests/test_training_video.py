"""Model, training and video tests (skipped without PyTorch / torchvision / OpenCV)."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
cv2 = pytest.importorskip("cv2")

from roadanomaly.frames import extract  # noqa: E402
from roadanomaly.models import BACKBONES, build_model, main_logits  # noqa: E402
from roadanomaly.predict_video import main as video_main  # noqa: E402
from roadanomaly.train import main as train_main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", list(BACKBONES))
def test_backbones_build_with_new_head(name):
    m = build_model(name, 4, pretrained=False).eval()
    size = 299 if name == "inception_v3" else 64
    out = main_logits(m(torch.randn(2, 3, size, size)))
    assert out.shape == (2, 4)


def _video(path, n=20, color=(0, 0, 255)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(n):
        frame = np.zeros((64, 64, 3), np.uint8)
        frame[:] = color
        frame[i % 64, :] = 255
        w.write(frame)
    w.release()


def test_frames_train_and_video(tmp_path):
    colors = {"accident": (0, 0, 255), "car_fire": (0, 128, 255), "fighting": (255, 0, 0), "snatching": (0, 255, 0)}
    for cls, col in colors.items():
        (tmp_path / "videos" / cls).mkdir(parents=True)
        _video(tmp_path / "videos" / cls / f"{cls}.mp4", 24, col)
        assert extract(tmp_path / "videos" / cls / f"{cls}.mp4", tmp_path / "frames" / cls, every=2) == 12

    run = tmp_path / "run"
    train_main(["--data", str(tmp_path / "frames"), "--config", str(ROOT / "configs/tiny.yaml"),
                "--workers", "0", "--out", str(run)])
    assert json.loads((run / "test_metrics.json").read_text())["accuracy"] >= 0.0

    video_main(["--run", str(run), "--input", str(tmp_path / "videos/fighting/fighting.mp4"),
                "--output", str(tmp_path / "out.mp4"), "--window", "4"])
    labels = (tmp_path / "out.labels.csv").read_text().splitlines()
    assert len(labels) == 25 and (tmp_path / "out.mp4").exists()
