from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.833, target=0.926)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def parse_order(text: str, did: str) -> list[int]:
    tokens = str(text).split()
    try:
        return [int(t) for t in tokens]
    except ValueError:
        fail(f"hội thoại {did}: cột order phải gồm các số nguyên cách nhau bởi dấu cách, nhận được {str(text)[:60]!r}.")


def pairwise_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    n = true.size
    dt = np.sign(true[:, None] - true[None, :])
    dp = np.sign(pred[:, None] - pred[None, :])
    inversions = int(np.triu(dt * dp < 0, k=1).sum())
    return 1.0 - inversions / (n * (n - 1) / 2)


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài B: độ chính xác thứ tự theo cặp (1 - I/M), trung bình theo hội thoại.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str, "order": str})

    sub = read_submission(args.pred, ["id", "order"], dtype={"id": str, "order": str}, keep_default_na=False)
    if (sub["id"].str.strip() == "").any():
        fail("có id rỗng.")
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}.")
    missing = set(gt["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(gt["id"])
    if missing:
        fail(f"thiếu {len(missing)} hội thoại, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")

    pred_map = dict(zip(sub["id"], sub["order"]))
    scores, exact, invalid = [], [], 0
    for did, order in zip(gt["id"], gt["order"]):
        true = np.array([int(t) for t in order.split()])
        pred = parse_order(pred_map[did], did)
        n = true.size
        if len(pred) != n or sorted(pred) != list(range(n)):
            invalid += 1  # không phải hoán vị hợp lệ của 0..n-1 -> 0 điểm cho hội thoại này
            scores.append(0.0)
            exact.append(0.0)
            continue
        pred = np.array(pred)
        scores.append(pairwise_accuracy(true, pred))
        exact.append(float((pred == true).all()))
    if invalid:
        sys.stderr.write(f"CẢNH BÁO: {invalid} hội thoại có order không phải hoán vị hợp lệ — tính 0 điểm.\n")
    acc = float(np.mean(scores))
    report(task_name(__file__, args.split), {"pairwise_acc": acc, "exact_match": float(np.mean(exact)), "n_invalid": float(invalid)},
           THRESHOLD.to_points(acc))


if __name__ == "__main__":
    main()
