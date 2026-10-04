"""Chấm Bài B: ROC-AUC trên tập test (điểm), kèm Brier score để tham khảo.

Dùng: python score.py [--pred ./submission.csv] [--data ./data] [--split public|private]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.68, target=0.79)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str})

    sub = read_submission(args.pred, ["id", "prob"], dtype={"id": str})
    if sub["id"].isna().any():
        fail("có id rỗng.")
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}.")
    missing = set(gt["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(gt["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")
    prob = pd.to_numeric(sub["prob"], errors="coerce")
    if prob.isna().any() or not np.isfinite(prob).all():
        fail("cột prob có giá trị rỗng/NaN hoặc không phải số.")
    if ((prob < 0) | (prob > 1)).any():
        fail("prob phải nằm trong [0, 1].")

    merged = gt.merge(sub.assign(prob=prob), on="id")
    auc = float(roc_auc_score(merged["vo_no"], merged["prob"]))
    brier = float(brier_score_loss(merged["vo_no"], merged["prob"]))
    report(task_name(__file__, args.split), {"roc_auc": auc, "brier": brier}, THRESHOLD.to_points(auc))


if __name__ == "__main__":
    main()
