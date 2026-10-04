"""Baseline bài A: nearest centroid trên pixel thô (sau chuẩn hoá từng ảnh).

Với mỗi episode: chuẩn hoá mỗi ảnh (đảo mực, căn trọng tâm, mean 0 / norm 1), tính tâm của 5 ảnh support
mỗi lớp, gán query cho tâm gần nhất (khoảng cách Euclid). Không dùng tập train.

Dùng: python baseline.py [--data ./data] [--split public|private] [--out ./submission.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

HERE = Path(__file__).resolve().parent


def normalize(x: np.ndarray) -> np.ndarray:
    """Chuẩn hoá: đảo về 'mực' theo nền từng ảnh, căn trọng tâm mực về giữa, trừ trung bình, chia chuẩn L2."""
    x = x.astype(np.float32)
    flat = x.reshape(len(x), -1)
    bg, lo = np.median(flat, 1)[:, None, None], flat.min(1)[:, None, None]
    ink = np.clip((bg - x) / (bg - lo + 1e-3), 0, 1)
    yy, xx = np.mgrid[0:x.shape[1], 0:x.shape[2]]
    out = np.empty_like(ink)
    for i, im in enumerate(ink):
        w = im**2
        m = w.sum() + 1e-6
        cy, cx = (w * yy).sum() / m, (w * xx).sum() / m
        out[i] = ndimage.shift(im, ((x.shape[1] - 1) / 2 - cy, (x.shape[2] - 1) / 2 - cx), order=1, mode="constant")
    out = out.reshape(len(x), -1)
    out -= out.mean(1, keepdims=True)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(test_dir / "data.npz", args.split)
    d = np.load(test_dir / "data.npz")
    s_x, s_ep, s_y = normalize(d["support_images"]), d["support_episode"], d["support_label"]
    q_x, q_ep, q_id = normalize(d["query_images"]), d["query_episode"], d["query_id"]

    ids, preds = [], []
    for e in np.unique(q_ep):
        sm, qm = s_ep == e, q_ep == e
        labels = np.unique(s_y[sm])
        cent = np.stack([s_x[sm][s_y[sm] == c].mean(0) for c in labels])
        dist = ((q_x[qm][:, None, :] - cent[None]) ** 2).sum(-1)
        preds.extend(labels[dist.argmin(1)].tolist())
        ids.extend(q_id[qm].tolist())
    pd.DataFrame({"query_id": ids, "label": preds}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
