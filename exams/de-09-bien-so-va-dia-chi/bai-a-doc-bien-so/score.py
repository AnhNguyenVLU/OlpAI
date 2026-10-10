from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.44, target=0.95)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên
MAX_LEN = 16
STRIP = re.compile(r"[\s\-.|]")
VALID = re.compile(r"[0-9A-Z]+")


def normalize(s: str) -> str:
    s = STRIP.sub("", str(s))
    return s.upper() if s.isascii() else s


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài A: trung bình 1 - min(1, Levenshtein / độ dài đáp án), kèm tỉ lệ đúng cả biển.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype=str, keep_default_na=False)

    sub = read_submission(args.pred, ["id", "plate"], dtype=str, keep_default_na=False)
    if (sub["id"].str.strip() == "").any():
        fail("có id rỗng.")
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}.")
    missing = set(gt["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(gt["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")
    pred = sub.set_index("id")["plate"].map(normalize)
    bad = pred[(pred == "") | ~pred.map(lambda s: bool(VALID.fullmatch(s)))]
    if len(bad):
        fail(f"id={bad.index[0]}: biển số dự đoán '{sub.set_index('id').at[bad.index[0], 'plate']}' rỗng hoặc chứa ký tự "
             "ngoài 0-9, A-Z (sau khi bỏ khoảng trắng, '-', '.', '|').")
    too_long = pred[pred.str.len() > MAX_LEN]
    if len(too_long):
        fail(f"id={too_long.index[0]}: biển số dự đoán dài hơn {MAX_LEN} ký tự.")

    truth = gt.set_index("id")["plate"]
    scores, exact = [], []
    for iid, t in truth.items():
        p = pred.at[iid]
        scores.append(1.0 - min(1.0, levenshtein(p, t) / len(t)))
        exact.append(p == t)
    score = float(np.mean(scores))
    report(task_name(__file__, args.split), {"score": score, "exact_match": float(np.mean(exact))}, THRESHOLD.to_points(score))


if __name__ == "__main__":
    main()
