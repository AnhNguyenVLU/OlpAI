from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

HERE = Path(__file__).resolve().parent
THRESHOLD = Threshold(baseline=0.642, target=0.775)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def load_pred(path: Path, truth: pd.DataFrame) -> pd.Series:
    pred = read_submission(path, ["query_id", "label"], dtype={"query_id": str})
    if pred["query_id"].isna().any() or pred["label"].isna().any():
        fail("có giá trị rỗng/NaN.")
    pred["query_id"] = pred["query_id"].str.strip()
    if pred["query_id"].duplicated().any():
        fail(f"trùng query_id, vd {pred.loc[pred['query_id'].duplicated(), 'query_id'].iloc[0]}.")
    missing = set(truth["query_id"]) - set(pred["query_id"])
    if missing:
        fail(f"thiếu {len(missing)} query_id, vd {sorted(missing)[0]}.")
    extra = set(pred["query_id"]) - set(truth["query_id"])
    if extra:
        fail(f"có {len(extra)} query_id lạ, vd {sorted(extra)[0]}.")
    lab = pd.to_numeric(pred["label"], errors="coerce")
    if lab.isna().any() or (lab % 1 != 0).any() or (lab < 0).any() or (lab > 19).any():
        fail("label phải là số nguyên trong [0, 19].")
    return lab.astype(int).set_axis(pred["query_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Chấm bài A: accuracy trung bình theo episode (mean over episodes của accuracy trên query).")
    ap.add_argument("--pred", type=Path, default=Path("submission.csv"))
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(test_dir / "answers.csv", args.split)
    truth = pd.read_csv(test_dir / "answers.csv", dtype={"query_id": str})
    pred = load_pred(args.pred, truth)
    correct = (pred.loc[truth["query_id"]].to_numpy() == truth["label"].to_numpy()).astype(float)
    ep_acc = pd.Series(correct).groupby(truth["episode"].to_numpy()).mean()
    acc = float(ep_acc.mean())
    ci95 = float(1.96 * ep_acc.std(ddof=1) / np.sqrt(len(ep_acc))) if len(ep_acc) > 1 else 0.0
    report(task_name(__file__, args.split),
           {"mean_episode_accuracy": acc, "ci95": ci95, "n_episodes": float(len(ep_acc))},
           THRESHOLD.to_points(acc))


if __name__ == "__main__":
    main()
