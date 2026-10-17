from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def global_features(images: np.ndarray) -> np.ndarray:
    x = images.astype(np.float64) / 255.0
    x = x / ndi.gaussian_filter(x, (0, 8, 8), mode="reflect")  # bù chiếu sáng thô
    n, h, w = x.shape
    flat = x.reshape(n, -1)
    feats = [flat.mean(1), flat.std(1)] + list(np.percentile(flat, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9], axis=1))
    gx, gy = np.diff(x, axis=2), np.diff(x, axis=1)
    feats += [np.log((gx ** 2).mean((1, 2)) + 1e-8), np.log((gy ** 2).mean((1, 2)) + 1e-8)]
    spec = np.abs(np.fft.fft2(x - x.mean((1, 2), keepdims=True))) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij")
    rad = np.hypot(fy, fx)
    for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 20), (20, 33)):
        sel = (rad >= lo) & (rad < hi)
        feats.append(np.log(spec[:, sel].mean(1) + 1e-8))
    return np.stack(feats, 1)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Baseline Bài B: đặc trưng toàn ảnh + khoảng cách Mahalanobis tới ảnh bình thường cùng dòng.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "train.npz")
    require_data(test_dir / "test.npz", args.split)
    with np.load(args.data / "train" / "train.npz", allow_pickle=False) as z:
        tr_img, tr_line, tr_kind = z["images"], z["mau_gach"], z["loai_loi"]
    with np.load(test_dir / "test.npz", allow_pickle=False) as z:
        te_img, te_line, te_ids = z["images"], z["mau_gach"], z["ids"]

    normal = tr_kind == "khong_loi"
    f_tr, f_te = global_features(tr_img), global_features(te_img)
    score = np.zeros(len(te_img))
    for line in np.unique(te_line):
        ref = f_tr[normal & (tr_line == line)]
        mu, sd = ref.mean(0), ref.std(0) + 1e-9
        cov = LedoitWolf().fit((ref - mu) / sd)
        sel = te_line == line
        score[sel] = np.sqrt(cov.mahalanobis((f_te[sel] - mu) / sd))
    h, w = te_img.shape[1:]
    pd.DataFrame({"id": te_ids, "score": score.round(6), "x": w / 2, "y": h / 2}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
