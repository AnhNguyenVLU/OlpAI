from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

# Mốc điểm thô đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên.
BASELINE = 0.583
TARGET = 0.94


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Chấm bài B (nhận dạng ký hiệu tay): macro-F1 trên 30 lớp, kèm accuracy tham khảo.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    ans_path = args.data / ("test_public" if args.split == "public" else "_private/test") / "answers.csv"
    require_data(ans_path, args.split)
    ans = pd.read_csv(ans_path, dtype=str)
    labels = set(pd.read_csv(args.data / "train" / "label_names.csv", dtype=str)["label"])

    sub = read_submission(args.pred, ["id", "label"], dtype=str)
    sub["id"] = sub["id"].str.strip()
    sub["label"] = sub["label"].str.strip()
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].head(5).tolist()}")
    if sub["label"].isna().any():
        fail(f"{int(sub['label'].isna().sum())} nhãn rỗng/NaN")
    missing = set(ans["id"]) - set(sub["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[:5]}")
    extra = set(sub["id"]) - set(ans["id"])
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[:5]}")
    bad = set(sub["label"]) - labels
    if bad:
        fail(f"nhãn không hợp lệ: {sorted(bad)[:5]} (xem train/label_names.csv)")

    pred = sub.set_index("id").loc[ans["id"], "label"].to_numpy()
    y = ans["label"].to_numpy()
    metrics = {"macro_f1": float(f1_score(y, pred, average="macro")), "accuracy": float(accuracy_score(y, pred))}
    points = Threshold(baseline=BASELINE, target=TARGET).to_points(metrics["macro_f1"])
    report(task_name(__file__, args.split), metrics, points)


if __name__ == "__main__":
    main()
