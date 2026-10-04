from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

HERE = Path(__file__).parent
# Đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên
THRESHOLD = Threshold(baseline=0.755, target=0.875)


def split_labels(s: str) -> list[str]:
    return [t.strip() for t in str(s).split("|") if t.strip()]


def to_matrix(series: pd.Series, labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(series), len(labels)), dtype=bool)
    for r, s in enumerate(series):
        for l in split_labels(s):
            m[r, idx[l]] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Chấm Bài B — trung bình micro-F1 và macro-F1 trên 15 nhãn ý định.")
    ap.add_argument("--pred", type=Path, default=HERE / "submission.csv")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    split_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(split_dir / "answers.csv", args.split)
    require_data(args.data / "train" / "labels.csv", "public")
    labels = pd.read_csv(args.data / "train" / "labels.csv")["label"].tolist()
    truth = pd.read_csv(split_dir / "answers.csv", keep_default_na=False, dtype=str)
    pred = read_submission(args.pred, ["id", "labels"], keep_default_na=False, dtype=str)
    pred["id"] = pred["id"].str.strip()

    if (pred["id"] == "").any():
        fail("có id rỗng.")
    dup = pred["id"].duplicated()
    if dup.any():
        fail(f"trùng id: {pred.loc[dup, 'id'].head(5).tolist()}.")
    missing = set(truth["id"]) - set(pred["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[:5]}.")
    extra = set(pred["id"]) - set(truth["id"])
    if extra:
        fail(f"có {len(extra)} id lạ không thuộc tập test, ví dụ {sorted(extra)[:5]}.")
    known = set(labels)
    bad = {l for s in pred["labels"] for l in split_labels(s) if l not in known}
    if bad:
        fail(f"nhãn không hợp lệ: {sorted(bad)[:5]} (xem train/labels.csv).")

    m = truth.merge(pred, on="id", suffixes=("_true", "_pred"))
    yt, yp = to_matrix(m["labels_true"], labels), to_matrix(m["labels_pred"], labels)
    tp = (yt & yp).sum(0).astype(float)
    fp = (~yt & yp).sum(0).astype(float)
    fn = (yt & ~yp).sum(0).astype(float)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1.0)
    denom = 2 * tp + fp + fn
    macro = float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)))
    metric = (micro + macro) / 2
    report(task_name(__file__, args.split), {"score": metric, "micro_f1": micro, "macro_f1": macro},
           THRESHOLD.to_points(metric))


if __name__ == "__main__":
    main()
