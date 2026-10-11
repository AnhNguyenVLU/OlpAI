from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

PATCH = 10  # cửa sổ được thu về 10x10 pixel xám
WINDOWS = (18, 30)  # hai cỡ cửa sổ vuông (pixel)
NEG_PER_IMAGE = 20


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def crop_feature(img: Image.Image, box) -> np.ndarray:
    patch = img.crop(tuple(int(round(v)) for v in box)).resize((PATCH, PATCH), Image.BILINEAR)
    return np.asarray(patch, np.float32).ravel() / 255.0


def windows(img: Image.Image, size: int):
    # mọi cửa sổ vuông cỡ `size`, bước size/PATCH pixel: thu nhỏ ảnh để cửa sổ thành PATCH x PATCH
    f = PATCH / size
    small = np.asarray(img.resize((max(PATCH, round(img.width * f)), max(PATCH, round(img.height * f))),
                                  Image.BILINEAR), np.float32) / 255.0
    view = np.lib.stride_tricks.sliding_window_view(small, (PATCH, PATCH))
    ny, nx = view.shape[:2]
    feats = view.reshape(ny * nx, PATCH * PATCH)
    ys, xs = np.mgrid[0:ny, 0:nx]
    x1, y1 = xs.ravel() / f, ys.ravel() / f
    return feats, np.stack([x1, y1, x1 + size, y1 + size], 1)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ov = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ov < thr]
    return np.array(keep, int)


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài A: cửa sổ trượt 2 cỡ + LogisticRegression trên patch xám 10x10 + NMS.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "answers.csv")
    require_data(test_dir / "images", args.split)
    ans = pd.read_csv(args.data / "train" / "answers.csv", dtype={"image_id": str})
    by_img = {k: g for k, g in ans.groupby("image_id")}

    xs, ys = [], []
    for path in sorted((args.data / "train" / "images").glob("*.jpg")):
        img = load_gray(path)
        g = by_img.get(path.stem)
        gt = g[["x1", "y1", "x2", "y2"]].to_numpy(float) if g is not None else np.zeros((0, 4))
        if g is not None:
            for (x1, y1, x2, y2), ign in zip(gt, g["ignore"].to_numpy()):
                if ign:
                    continue
                s = max(x2 - x1, y2 - y1)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                xs.append(crop_feature(img, (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)))
                ys.append(1)
        for _ in range(NEG_PER_IMAGE):
            s = WINDOWS[int(rng.integers(len(WINDOWS)))]
            x1, y1 = rng.uniform(0, img.width - s), rng.uniform(0, img.height - s)
            box = np.array([[x1, y1, x1 + s, y1 + s]])
            if len(gt) and iou_matrix(box, gt).max() >= 0.3:
                continue
            xs.append(crop_feature(img, box[0]))
            ys.append(0)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(np.array(xs), np.array(ys))

    rows = []
    for path in sorted((test_dir / "images").glob("*.jpg")):
        img = load_gray(path)
        feats, boxes = [], []
        for s in WINDOWS:
            f, b = windows(img, s)
            feats.append(f)
            boxes.append(b)
        feats, boxes = np.concatenate(feats), np.concatenate(boxes)
        prob = clf.predict_proba(feats)[:, 1]
        sel = prob > 0.5
        boxes, prob = boxes[sel], prob[sel]
        keep = nms(boxes, prob, 0.3)[:100]
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            rows.append((path.stem, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(float(prob[i]), 5)))
    pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
