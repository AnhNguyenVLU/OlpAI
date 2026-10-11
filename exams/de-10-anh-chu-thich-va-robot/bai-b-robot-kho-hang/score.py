from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.604, target=0.96)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Chấm Bài B: độ chính xác trung bình theo robot (6 robot, trọng số bằng nhau).")
    ap.add_argument("--pred", type=Path, default=Path("submission.csv"))
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str})

    sub = read_submission(args.pred, ["id", "hanh_dong"], dtype={"id": str})
    if sub["id"].isna().any():
        fail("có id rỗng.")
    sub["id"] = sub["id"].str.strip()
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}.")
    missing = set(gt["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(gt["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")
    act = pd.to_numeric(sub["hanh_dong"], errors="coerce")
    if act.isna().any() or not np.isfinite(act.to_numpy(float)).all() or (act % 1 != 0).any():
        fail("hanh_dong phải là số nguyên (không rỗng/NaN).")
    if ((act < 0) | (act > 5)).any():
        fail("hanh_dong phải thuộc {0, 1, 2, 3, 4, 5}.")

    pred = act.astype(int).set_axis(sub["id"])
    correct = pred.loc[gt["id"]].to_numpy() == gt["hanh_dong"].to_numpy()
    per_robot = pd.Series(correct).groupby(gt["robot"].to_numpy()).mean()
    score = float(per_robot.mean())
    metrics = {"mean_robot_accuracy": score, "overall_accuracy": float(correct.mean())}
    metrics.update({f"acc_robot_{int(r)}": float(v) for r, v in per_robot.items()})
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(score))


if __name__ == "__main__":
    main()
