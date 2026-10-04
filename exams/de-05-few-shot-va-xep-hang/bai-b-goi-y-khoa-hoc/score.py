from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

HERE = Path(__file__).resolve().parent
K = 10
THRESHOLD = Threshold(baseline=0.034, target=0.282)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def load_pred(path: Path, users: list[str], valid_courses: set[str]) -> dict[str, list[str]]:
    pred = read_submission(path, ["user_id", "course_ids"], dtype=str, keep_default_na=False)
    pred = pred.apply(lambda col: col.str.strip())
    if (pred["user_id"] == "").any() or (pred["course_ids"] == "").any():
        fail("có giá trị rỗng/NaN.")
    if pred["user_id"].duplicated().any():
        fail(f"trùng user_id, vd {pred.loc[pred['user_id'].duplicated(), 'user_id'].iloc[0]}.")
    missing = set(users) - set(pred["user_id"])
    if missing:
        fail(f"thiếu {len(missing)} user_id, vd {sorted(missing)[0]}.")
    extra = set(pred["user_id"]) - set(users)
    if extra:
        fail(f"có {len(extra)} user_id không thuộc tập test, vd {sorted(extra)[0]}.")
    out: dict[str, list[str]] = {}
    for u, s in zip(pred["user_id"], pred["course_ids"]):
        recs = s.split()
        if len(recs) != K:
            fail(f"user {u}: cần đúng {K} course_id, nhận {len(recs)}.")
        if len(set(recs)) != K:
            fail(f"user {u}: course_id bị lặp.")
        bad = [c for c in recs if c not in valid_courses]
        if bad:
            fail(f"user {u}: course_id không tồn tại: {bad[0]}.")
        out[u] = recs
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Chấm bài B: NDCG@10 trung bình theo người dùng (điểm chính), kèm Recall@10.")
    ap.add_argument("--pred", type=Path, default=Path("submission.csv"))
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(test_dir / "answers.csv", args.split)
    require_data(args.data / "train" / "courses.csv", "public")
    users = pd.read_csv(test_dir / "test_users.csv", dtype=str)["user_id"].tolist()
    truth = pd.read_csv(test_dir / "answers.csv", dtype=str).groupby("user_id")["course_id"].agg(set).to_dict()
    courses = set(pd.read_csv(args.data / "train" / "courses.csv", dtype=str)["course_id"])
    pred = load_pred(args.pred, users, courses)

    disc = 1.0 / np.log2(np.arange(2, K + 2))
    ndcg, recall = [], []
    for u in users:
        rel = truth.get(u, set())
        hits = np.array([c in rel for c in pred[u]], dtype=float)
        idcg = disc[: min(len(rel), K)].sum()
        ndcg.append((hits * disc).sum() / idcg if idcg > 0 else 0.0)
        recall.append(hits.sum() / len(rel) if rel else 0.0)
    m = float(np.mean(ndcg))
    report(task_name(__file__, args.split),
           {"ndcg@10": m, "recall@10": float(np.mean(recall)), "n_users": float(len(users))},
           THRESHOLD.to_points(m))


if __name__ == "__main__":
    main()
