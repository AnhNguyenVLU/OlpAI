from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

CHAR_W, CHAR_H = 10, 16


def otsu(a: np.ndarray) -> int:
    p = np.bincount(a.ravel(), minlength=256).astype(float)
    p /= p.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-9)
    return int(np.argmax(between))


def segment(a: np.ndarray) -> list[list[np.ndarray]]:
    # tách ký tự theo cách đơn giản của baseline (mô tả trong statement.md)
    lab, n = ndimage.label(a < otsu(a), structure=np.ones((3, 3)))
    boxes = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 8 <= h <= 30 and 2 <= w <= 1.2 * h:
            boxes.append(sl)
    if not boxes:
        return []
    med = np.median([b[0].stop - b[0].start for b in boxes])
    boxes = [b for b in boxes if abs((b[0].stop - b[0].start) - med) <= 0.3 * med]
    cy = np.array([(b[0].start + b[0].stop) / 2 for b in boxes])
    order = np.argsort(cy)
    gaps = np.diff(cy[order])
    lines = [list(order)]
    if len(gaps) and gaps.max() > 0.8 * med:
        k = int(np.argmax(gaps)) + 1
        lines = [list(order[:k]), list(order[k:])]
    out = []
    for line in lines:
        line.sort(key=lambda i: boxes[i][1].start)
        crops = []
        for i in line:
            c = Image.fromarray(a[boxes[i]]).resize((CHAR_W, CHAR_H), Image.Resampling.BILINEAR)
            x = np.asarray(c, dtype=np.float32).ravel()
            crops.append((x - x.mean()) / (x.std() + 1e-6))
        out.append(crops)
    return out


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài A: ngưỡng Otsu + thành phần liên thông, phân loại từng ký tự bằng LogisticRegression trên điểm ảnh.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "labels.csv")
    require_data(test_dir / "images", args.split)
    train = pd.read_csv(args.data / "train" / "labels.csv", dtype=str)

    xs, ys = [], []
    for iid, plate in zip(train["id"], train["plate"]):
        a = np.asarray(Image.open(args.data / "train" / "images" / f"{iid}.png").convert("L"))
        crops = [c for line in segment(a) for c in line]
        if len(crops) == len(plate):  # chỉ dùng ảnh tách được đúng số ký tự
            xs += crops
            ys += list(plate)
    clf = LogisticRegression(C=0.1, max_iter=1000)
    clf.fit(np.stack(xs), ys)

    rows = []
    for path in sorted((test_dir / "images").glob("*.png")):
        a = np.asarray(Image.open(path).convert("L"))
        crops = [c for line in segment(a) for c in line]
        pred = "".join(clf.predict(np.stack(crops))) if crops else "0"
        rows.append({"id": path.stem, "plate": pred})
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
