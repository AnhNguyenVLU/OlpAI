from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

HERE = Path(__file__).parent
QUANTILES = (0.1, 0.5, 0.9)
COLS = ["p10", "p50", "p90"]
KEY = ["station_id", "timestamp"]
# Đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên
THRESHOLD = Threshold(baseline=0.141, target=0.045)


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    d = y - pred
    return np.maximum(q * d, (q - 1) * d)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chấm Bài A — pinball loss trung bình (P10/P50/P90) chuẩn hoá theo quy mô trạm.")
    ap.add_argument("--pred", type=Path, default=HERE / "submission.csv")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    split_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(split_dir / "answers.csv", args.split)
    truth = pd.read_csv(split_dir / "answers.csv", dtype={"timestamp": str})
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    pred = read_submission(args.pred, KEY + COLS, dtype={"station_id": str, "timestamp": str})

    pred["timestamp"] = pd.to_datetime(pred["timestamp"], format="%Y-%m-%d %H:%M", errors="coerce")
    if pred["timestamp"].isna().any():
        fail("timestamp không hợp lệ (cần định dạng 'YYYY-MM-DD HH:MM').")
    dup = pred.duplicated(KEY)
    if dup.any():
        fail(f"trùng khoá (station_id, timestamp): {int(dup.sum())} dòng.")
    for c in COLS:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")
    vals = pred[COLS].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        fail("có giá trị trống/NaN/không phải số/vô hạn trong p10/p50/p90.")
    m = truth.merge(pred, on=KEY, how="left", indicator=True)
    n_miss = int((m["_merge"] == "left_only").sum())
    if n_miss:
        fail(f"thiếu dự báo cho {n_miss} dòng (station_id, timestamp) của test.csv.")
    if len(pred) != len(truth):
        fail(f"file nộp có {len(pred)} dòng, cần đúng {len(truth)} dòng như test.csv (có dòng lạ).")

    qs = np.sort(m[COLS].to_numpy(dtype=float), axis=1)
    y = m["load_mw"].to_numpy(dtype=float)
    m["loss"] = np.mean([pinball(y, qs[:, i], q) for i, q in enumerate(QUANTILES)], axis=0)
    per = m.groupby("station_id").agg(loss=("loss", "mean"), scale=("load_mw", "mean"))
    metric = float((per["loss"] / per["scale"]).mean())
    cover = float(np.mean((y >= qs[:, 0]) & (y <= qs[:, 2])))
    report(task_name(__file__, args.split), {"scaled_pinball": metric, "coverage_p10_p90": cover},
           THRESHOLD.to_points(metric))


if __name__ == "__main__":
    main()
