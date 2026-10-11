from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.19, target=0.78)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài A: macro-AP theo loài (điểm), kèm micro-AP để tham khảo.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str})
    species = [c for c in gt.columns if c != "id"]

    sub = read_submission(args.pred, ["id", *species], dtype=str)  # đọc mọi ô dạng chuỗi rồi tự kiểm tra kiểu số
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
    prob = sub[species].apply(pd.to_numeric, errors="coerce").astype(float)  # chữ, True/False, ô rỗng -> NaN
    bad = prob.columns[~np.isfinite(prob.to_numpy()).all(axis=0)].tolist()
    if bad:
        fail(f"có giá trị rỗng, NaN, inf hoặc không phải số ở cột {bad[:3]} ({len(bad)} cột).")
    out_of_range = prob.columns[((prob < 0) | (prob > 1)).any(axis=0).to_numpy()].tolist()
    if out_of_range:
        fail(f"xác suất phải nằm trong [0, 1]; vi phạm ở cột {out_of_range[:3]} ({len(out_of_range)} cột).")

    pred = pd.concat([sub[["id"]], prob], axis=1).set_index("id").loc[gt["id"], species].to_numpy(float)
    true = gt[species].to_numpy(int)
    has_pos = true.sum(axis=0) > 0
    ap_per = [average_precision_score(true[:, k], pred[:, k]) for k in np.flatnonzero(has_pos)]
    macro_ap = float(np.mean(ap_per))
    micro_ap = float(average_precision_score(true.ravel(), pred.ravel()))
    report(task_name(__file__, args.split), {"macro_ap": macro_ap, "micro_ap": micro_ap, "n_species_scored": float(has_pos.sum())},
           THRESHOLD.to_points(macro_ap))


if __name__ == "__main__":
    main()
