from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.214, target=0.330)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm bài B — Gợi ý hợp tác nghiên cứu (MRR trên 100 ứng viên mỗi truy vấn).")
    ap.add_argument("--pred", type=Path, default=Path("submission.csv"))
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gold_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gold_dir / "answers.csv", args.split)
    cand = pd.read_csv(gold_dir / "candidates.csv", dtype=str)
    ans = pd.read_csv(gold_dir / "answers.csv", dtype=str)
    pred = read_submission(args.pred, ["query_id", "candidate_id", "score"],
                           dtype={"query_id": str, "candidate_id": str})
    pred["query_id"] = pred["query_id"].str.strip()
    pred["candidate_id"] = pred["candidate_id"].str.strip()
    if pred[["query_id", "candidate_id"]].isna().any().any():
        fail("có query_id/candidate_id rỗng.")
    dup = pred.duplicated(["query_id", "candidate_id"])
    if dup.any():
        r = pred[dup].iloc[0]
        fail(f"cặp bị trùng: {r.query_id},{r.candidate_id}.")
    score = pd.to_numeric(pred["score"], errors="coerce")
    if score.isna().any() or not np.isfinite(score.to_numpy(float)).all():
        fail("score phải là số hữu hạn (không rỗng/NaN/inf).")
    pred["score"] = score.astype(float)
    m = cand.merge(pred, on=["query_id", "candidate_id"], how="left", indicator=True)
    miss = m["_merge"] != "both"
    if miss.any():
        r = m[miss].iloc[0]
        fail(f"thiếu {int(miss.sum())} cặp ứng viên, ví dụ {r.query_id},{r.candidate_id}.")
    if len(pred) != len(cand):
        fail(f"có {len(pred) - len(cand)} cặp không thuộc danh sách ứng viên.")

    pos = set(zip(ans["query_id"], ans["candidate_id"]))
    m["pos"] = [(q, c) in pos for q, c in zip(m["query_id"], m["candidate_id"])]
    rr, hits = [], []
    for _, g in m.groupby("query_id", sort=False):
        s = g["score"].to_numpy()
        is_pos = g["pos"].to_numpy()
        neg = np.sort(s[~is_pos])
        hi = np.searchsorted(neg, s[is_pos], "right")
        lo = np.searchsorted(neg, s[is_pos], "left")
        best = float(np.min(1 + (len(neg) - hi) + 0.5 * (hi - lo)))
        rr.append(1.0 / best)
        hits.append(best <= 10)
    rr_arr = np.array(rr)
    mrr = float(rr_arr.mean())
    metrics = {"mrr": mrr, "hits_at_10": float(np.mean(hits)),
               "mrr_ci95": float(1.96 * rr_arr.std(ddof=1) / np.sqrt(len(rr_arr)))}
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(mrr))


if __name__ == "__main__":
    main()
