from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

H = W = 64
N_CLASSES = 4
K_COUNT = 3.0
THRESHOLD = Threshold(baseline=0.31, target=0.57)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def rle_decode(s: str, iid: str) -> np.ndarray:
    labels, lengths = [], []
    for tok in str(s).split():
        parts = tok.split(":")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            fail(f"id={iid}: token RLE không hợp lệ '{tok}' (cần dạng nhan:do_dai)")
        lab, n = int(parts[0]), int(parts[1])
        if lab >= N_CLASSES or n <= 0:
            fail(f"id={iid}: nhãn phải trong 0..{N_CLASSES - 1} và độ dài > 0 ('{tok}')")
        labels.append(lab)
        lengths.append(n)
    if sum(lengths) != H * W:
        fail(f"id={iid}: tổng độ dài RLE = {sum(lengths)}, cần {H * W}")
    return np.repeat(np.array(labels, np.uint8), lengths).reshape(H, W)


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài A: score = 0.7 * mIoU + 0.3 * (1 - min(1, MAE_count / K)), K = 3.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt_counts = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str}).set_index("id")["count"]
    gt_ids = list(gt_counts.index)
    gt_masks = [np.asarray(Image.open(gt_dir / "masks" / f"{i}.png")) for i in gt_ids]

    sub = read_submission(args.pred, ["id", "mask", "count"], dtype={"id": str, "mask": str})
    if sub[["id", "mask", "count"]].isna().any().any():
        fail("có giá trị rỗng/NaN")
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}")
    missing = set(gt_ids) - set(sub["id"])
    extra = set(sub["id"]) - set(gt_ids)
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}")
    counts = pd.to_numeric(sub["count"], errors="coerce")
    if counts.isna().any() or not np.isfinite(counts).all():
        fail("cột count phải là số")
    sub = sub.assign(count=counts).set_index("id")

    conf = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for iid, gt in zip(gt_ids, gt_masks):
        pred = rle_decode(sub.at[iid, "mask"], iid)
        conf += np.bincount(gt.ravel().astype(np.int64) * N_CLASSES + pred.ravel(), minlength=N_CLASSES**2).reshape(
            N_CLASSES, N_CLASSES
        )
    inter = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    miou = float(iou.mean())
    mae = float(np.abs(sub.loc[gt_ids, "count"].to_numpy(float) - gt_counts.loc[gt_ids].to_numpy(float)).mean())
    score = 0.7 * miou + 0.3 * (1 - min(1.0, mae / K_COUNT))

    metrics = {"score": score, "mIoU": miou, "MAE_count": mae}
    metrics.update({f"IoU_{name}": float(v) for name, v in zip(["ruong", "kenh", "nha", "cay"], iou)})
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(score))


if __name__ == "__main__":
    main()
