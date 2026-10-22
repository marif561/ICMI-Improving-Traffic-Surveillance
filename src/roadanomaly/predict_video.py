"""Label a surveillance video frame by frame with rolling prediction and write an annotated copy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from PIL import Image

from .data import build_transforms
from .models import build_model, main_logits
from .rolling import RollingPredictor


def load_run(run_dir: str | Path, device: str):
    run = Path(run_dir)
    meta = json.loads((run / "meta.json").read_text())
    model = build_model(meta["backbone"], len(meta["classes"]), pretrained=False, dropout=meta.get("dropout", 0.5))
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    return model.to(device).eval(), meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="folder written by roadanomaly.train")
    ap.add_argument("--input", required=True, help="input video")
    ap.add_argument("--output", default=None, help="annotated output video (.mp4)")
    ap.add_argument("--window", type=int, default=32, help="frames averaged by rolling prediction")
    ap.add_argument("--every", type=int, default=1, help="classify every N-th frame")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_run(args.run, device)
    classes, tf = meta["classes"], build_transforms(meta["image_size"], {}, train=False)
    rolling = RollingPredictor(args.window)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer, rows, i, label, prob = None, [], 0, "", 0.0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every == 0:
                x = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                p = torch.softmax(main_logits(model(x)), -1)[0].cpu().numpy()
                avg = rolling.update(p)
                label, prob = classes[int(avg.argmax())], float(avg.max())
                rows.append({"frame": i, "frame_label": classes[int(p.argmax())], "smoothed_label": label,
                             "smoothed_prob": round(prob, 4)})
            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                cv2.putText(frame, f"{label} ({prob:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                writer.write(frame)
            i += 1
    cap.release()
    if writer:
        writer.release()

    csv_path = Path(args.output or args.input).with_suffix(".labels.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "frame_label", "smoothed_label", "smoothed_prob"])
        w.writeheader()
        w.writerows(rows)
    print(f"{i} frames processed · labels written to {csv_path}")


if __name__ == "__main__":
    main()
