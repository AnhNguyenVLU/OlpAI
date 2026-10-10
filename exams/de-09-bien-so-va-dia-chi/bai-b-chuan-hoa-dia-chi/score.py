from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.930, target=0.995)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên
FIELDS = ["so_nha", "duong", "xa_moi", "tinh_moi"]
MAX_LEN = 200


def canon(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài B: độ chính xác khớp chính xác trung bình trên 4 trường so_nha, duong, xa_moi, tinh_moi.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype=str, keep_default_na=False)

    sub = read_submission(args.pred, ["id"] + FIELDS, dtype=str, keep_default_na=False)
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
    for f in FIELDS:
        too_long = sub[sub[f].str.len() > MAX_LEN]
        if len(too_long):
            fail(f"id={too_long['id'].iloc[0]}: trường {f} dài hơn {MAX_LEN} ký tự.")

    sub = sub.set_index("id").loc[gt["id"]]
    truth = gt.set_index("id")
    metrics = {}
    hits = []
    for f in FIELDS:
        ok = np.array([canon(a) == canon(b) for a, b in zip(sub[f], truth[f])])
        metrics[f"acc_{f}"] = float(ok.mean())
        hits.append(ok)
    score = float(np.mean([metrics[f"acc_{f}"] for f in FIELDS]))
    metrics = {"score": score, **metrics, "full_address_acc": float(np.all(hits, axis=0).mean())}
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(score))


if __name__ == "__main__":
    main()
