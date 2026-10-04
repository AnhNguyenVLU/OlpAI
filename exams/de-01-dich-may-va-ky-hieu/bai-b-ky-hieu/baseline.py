"""Baseline bài B: thống kê mean/std theo từng keypoint (bỏ qua NaN) + LogisticRegression.

Dùng: python baseline.py --data data --out submission.csv [--split public|private]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def load(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    z = np.load(path, allow_pickle=False)
    X, lengths = z["X"].astype(np.float32), z["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return [X[o:o + n] for o, n in zip(offsets, lengths)], z["ids"]


def features(seqs: list[np.ndarray]) -> np.ndarray:
    feats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # điểm NaN ở mọi khung
        for s in seqs:
            flat = s.reshape(len(s), -1)
            feats.append(np.concatenate([np.nanmean(flat, 0), np.nanstd(flat, 0)]))
    F = np.array(feats)
    return np.nan_to_num(F, nan=0.0)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train", "public")
    require_data(test_dir, args.split)
    tr_seqs, tr_ids = load(args.data / "train" / "data.npz")
    ans = pd.read_csv(args.data / "train" / "answers.csv").set_index("id").loc[tr_ids, "label"].to_numpy()
    te_seqs, te_ids = load(test_dir / "data.npz")

    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
    with threadpool_limits(1):  # bài toán nhỏ: BLAS đa luồng chỉ làm chậm
        model.fit(features(tr_seqs), ans)
        pred = model.predict(features(te_seqs))
    pd.DataFrame({"id": te_ids, "label": pred}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
