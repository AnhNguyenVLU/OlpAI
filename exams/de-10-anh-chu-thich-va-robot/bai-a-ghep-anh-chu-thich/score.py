from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

K = 20
THRESHOLD = Threshold(baseline=0.237, target=0.99)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Chấm Bài A: MRR của ảnh đúng trong thứ hạng 20 ứng viên (kèm R@1).")
    ap.add_argument("--pred", type=Path, default=Path("submission.csv"))
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    queries = pd.read_csv(gt_dir / "queries.csv", dtype=str)
    answers = pd.read_csv(gt_dir / "answers.csv", dtype=str)
    cands = {q: c.split() for q, c in zip(queries["query_id"], queries["candidates"])}
    pos = dict(zip(answers["query_id"], answers["image_id"]))

    sub = read_submission(args.pred, ["query_id", "ranking"], dtype=str, keep_default_na=False)
    sub = sub.apply(lambda col: col.str.strip())
    if (sub["query_id"] == "").any() or (sub["ranking"] == "").any():
        fail("có query_id hoặc ranking rỗng.")
    if sub["query_id"].duplicated().any():
        fail(f"trùng query_id: {sub.loc[sub['query_id'].duplicated(), 'query_id'].iloc[0]}.")
    missing = set(cands) - set(sub["query_id"])
    extra = set(sub["query_id"]) - set(cands)
    if missing:
        fail(f"thiếu {len(missing)} query_id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} query_id lạ, ví dụ {sorted(extra)[0]}.")

    rr = []
    for q, ranking in zip(sub["query_id"], sub["ranking"]):
        ids = ranking.split()
        if len(ids) != K:
            fail(f"truy vấn {q}: ranking phải gồm đúng {K} image_id (nhận {len(ids)}).")
        if len(set(ids)) != K:
            fail(f"truy vấn {q}: ranking có image_id bị lặp.")
        if set(ids) != set(cands[q]):
            bad = sorted(set(ids) - set(cands[q]))[0]
            fail(f"truy vấn {q}: {bad} không thuộc 20 ứng viên của truy vấn.")
        rr.append(1.0 / (ids.index(pos[q]) + 1))
    rr_arr = np.array(rr)
    mrr = float(rr_arr.mean())
    metrics = {"mrr": mrr, "recall_at_1": float((rr_arr == 1.0).mean()),
               "mrr_ci95": float(1.96 * rr_arr.std(ddof=1) / np.sqrt(len(rr_arr))) if len(rr_arr) > 1 else 0.0,
               "n_queries": float(len(rr_arr))}
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(mrr))


if __name__ == "__main__":
    main()
