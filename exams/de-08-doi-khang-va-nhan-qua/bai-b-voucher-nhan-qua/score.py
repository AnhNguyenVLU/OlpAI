from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=2.5, target=14.6)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên
FRAC = 0.3
BOOTSTRAP = 200


def selection_weights(score: np.ndarray, frac: float) -> np.ndarray:
    # trọng số chọn top-k (k = round(frac * N)); các khách hoà điểm ở ngưỡng được chọn theo tỉ lệ (= kỳ vọng khi bốc ngẫu nhiên)
    n = len(score)
    k = int(round(frac * n))
    kth = np.sort(score)[::-1][k - 1]
    above = score > kth
    tie = score == kth
    w = above.astype(float)
    w[tie] = (k - above.sum()) / tie.sum()
    return w


def true_uplift(w: np.ndarray, tau: np.ndarray) -> float:
    return 100 * (float((w * tau).sum() / w.sum()) - float(tau.mean()))


def rct_uplift(w: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    # nan nếu nhóm được chọn không có người nhận hoặc không có người đối chứng (không ước lượng được)
    def diff(weight: np.ndarray) -> float:
        n_treated, n_control = (weight * t).sum(), (weight * (1 - t)).sum()
        if n_treated <= 0 or n_control <= 0:
            return float("nan")
        return (weight * t * y).sum() / n_treated - (weight * (1 - t) * y).sum() / n_control

    return 100 * float(diff(w) - diff(np.ones_like(w)))


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài B: uplift@30% (điểm phần trăm) của danh sách khách được chọn theo cột uplift.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str})

    sub = read_submission(args.pred, ["id", "uplift"], dtype={"id": str})
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
    uplift = pd.to_numeric(sub["uplift"], errors="coerce")
    if uplift.isna().any() or not np.isfinite(uplift).all():
        fail("cột uplift có giá trị rỗng/NaN/vô cực hoặc không phải số.")

    merged = gt.merge(sub.assign(uplift=uplift), on="id")
    score = merged["uplift"].to_numpy(float)
    w30, w10, w50 = (selection_weights(score, f) for f in (FRAC, 0.1, 0.5))
    if args.split == "private":
        tau = merged["tau"].to_numpy(float)
        metrics = {"uplift_at_30": true_uplift(w30, tau), "uplift_at_10": true_uplift(w10, tau),
                   "uplift_at_50": true_uplift(w50, tau)}
    else:
        t = merged["nhan_voucher"].to_numpy(float)
        y = merged["mua_hang"].to_numpy(float)
        est, est10, est50 = rct_uplift(w30, t, y), rct_uplift(w10, t, y), rct_uplift(w50, t, y)
        if not np.isfinite([est, est10, est50]).all():
            fail("trong nhóm khách được chọn (10%, 30% hoặc 50% điểm cao nhất) không có người nhận voucher hoặc không có "
                 "người đối chứng trong thí nghiệm của test_public, nên không ước lượng được uplift.")
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(BOOTSTRAP):
            i = rng.integers(0, len(t), len(t))
            boot.append(rct_uplift(w30[i], t[i], y[i]))
        metrics = {"uplift_at_30": est, "uplift_at_30_se": float(np.nanstd(boot)),
                   "uplift_at_10": est10, "uplift_at_50": est50}
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(metrics["uplift_at_30"]))


if __name__ == "__main__":
    main()
