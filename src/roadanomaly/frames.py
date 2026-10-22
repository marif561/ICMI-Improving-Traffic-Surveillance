"""Extract frames from labelled videos into an image-folder dataset.

Input layout:  <videos>/<class_name>/*.mp4
Output layout: <out>/<class_name>/<video>_<frame>.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}


def extract(video: Path, out_dir: Path, every: int = 5, size: int | None = None) -> int:
    cap = cv2.VideoCapture(str(video))
    out_dir.mkdir(parents=True, exist_ok=True)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            if size:
                frame = cv2.resize(frame, (size, size))
            cv2.imwrite(str(out_dir / f"{video.stem}_{i:06d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5, help="keep one frame out of every N")
    ap.add_argument("--size", type=int, default=None, help="optional square resize")
    args = ap.parse_args(argv)

    total = 0
    for cls_dir in sorted(p for p in Path(args.videos).iterdir() if p.is_dir()):
        for vid in sorted(cls_dir.iterdir()):
            if vid.suffix.lower() in VIDEO_EXT:
                n = extract(vid, Path(args.out) / cls_dir.name, args.every, args.size)
                total += n
                print(f"{cls_dir.name}/{vid.name}: {n} frames")
    print(f"done: {total} frames written to {args.out}")


if __name__ == "__main__":
    main()
