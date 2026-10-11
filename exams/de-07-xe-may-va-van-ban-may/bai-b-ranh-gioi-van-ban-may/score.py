from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

TAU = 100.0
THRESHOLD = Threshold(baseline=0.26, target=0.88)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài B: trung bình exp(-|p - t| / 100) theo ký tự, kèm sai số trung vị.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    require_data(gt_dir / "data.csv", args.split)
    lengths = pd.read_csv(gt_dir / "data.csv", dtype={"id": str}, keep_default_na=False).set_index("id")["text"].str.len()
    gt = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str}).set_index("id")["boundary_char_index"]

    sub = read_submission(args.pred, ["id", "boundary_char_index"], dtype={"id": str})
    if sub["id"].isna().any():
        fail("có id rỗng.")
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}.")
    missing = set(gt.index) - set(sub["id"])
    extra = set(sub["id"]) - set(gt.index)
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")
    val = pd.to_numeric(sub["boundary_char_index"], errors="coerce").to_numpy(float)
    if not np.isfinite(val).all():
        fail("cột boundary_char_index có giá trị rỗng/NaN hoặc không phải số.")
    if (val != np.round(val)).any():
        fail("boundary_char_index phải là số nguyên.")
    pred = pd.Series(val, index=sub["id"].to_numpy()).loc[gt.index]
    lim = lengths.loc[gt.index].to_numpy()
    if ((pred.to_numpy() < 0) | (pred.to_numpy() > lim)).any():
        k = int(np.flatnonzero((pred.to_numpy() < 0) | (pred.to_numpy() > lim))[0])
        fail(f"id={gt.index[k]}: boundary_char_index = {int(pred.iloc[k])} nằm ngoài [0, {int(lim[k])}] (độ dài văn bản).")
    err = np.abs(pred.to_numpy() - gt.to_numpy(float))
    score = float(np.exp(-err / TAU).mean())
    metrics = {"score": score, "median_abs_err": float(np.median(err)), "within_50": float((err <= 50).mean())}
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(score))


if __name__ == "__main__":
    main()
