from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

W, H = 128, 96
MAX_BOXES_PER_IMAGE = 100
COLS = ["image_id", "x1", "y1", "x2", "y2", "score"]
THRESHOLD = Threshold(baseline=0.07, target=0.79)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = iw * ih
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-12)


def average_precision(gt: dict, img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, thr: float) -> float:
    # gt: image_id -> (hộp Nx4, cờ ignore N). AP nội suy mọi điểm kiểu PASCAL VOC (2010+).
    npos = int(sum(int((~ig).sum()) for _, ig in gt.values()))
    if npos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    used = {k: np.zeros(len(v[0]), bool) for k, v in gt.items()}
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    keep = np.ones(len(order), bool)
    for r, d in enumerate(order):
        g, ig = gt.get(img[d], (np.zeros((0, 4)), np.zeros(0, bool)))
        if len(g) == 0:
            fp[r] = 1
            continue
        ov = iou_one(boxes[d], g)
        j = int(np.argmax(ov))
        if ov[j] >= thr:
            if ig[j]:
                keep[r] = False  # trùng hộp ignore: không tính
            elif not used[img[d]][j]:
                tp[r] = 1
                used[img[d]][j] = True
            else:
                fp[r] = 1  # phát hiện trùng lặp
        else:
            fp[r] = 1
    tp, fp = np.cumsum(tp[keep]), np.cumsum(fp[keep])
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài A: AP@0.5 (một lớp, nội suy mọi điểm kiểu VOC), kèm AP@0.75 để tham khảo.")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    require_data(gt_dir / "images", args.split)
    image_ids = sorted(p.stem for p in (gt_dir / "images").glob("*.jpg"))
    ans = pd.read_csv(gt_dir / "answers.csv", dtype={"image_id": str})
    gt = {i: (np.zeros((0, 4)), np.zeros(0, bool)) for i in image_ids}
    for iid, grp in ans.groupby("image_id"):
        gt[iid] = (grp[["x1", "y1", "x2", "y2"]].to_numpy(float), grp["ignore"].to_numpy(int).astype(bool))

    sub = read_submission(args.pred, COLS, dtype={"image_id": str})
    if sub["image_id"].isna().any():
        fail("có image_id rỗng.")
    bad = set(sub["image_id"]) - set(image_ids)
    if bad:
        fail(f"có {len(bad)} image_id không thuộc tập chấm, ví dụ {sorted(bad)[0]}.")
    num = sub[COLS[1:]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(num).all():
        fail("các cột x1, y1, x2, y2, score phải là số hữu hạn (không rỗng/NaN/inf).")
    if (num[:, 2] <= num[:, 0]).any() or (num[:, 3] <= num[:, 1]).any():
        r = int(np.flatnonzero((num[:, 2] <= num[:, 0]) | (num[:, 3] <= num[:, 1]))[0])
        fail(f"hộp không hợp lệ ở dòng dữ liệu {r + 1}: cần x2 > x1 và y2 > y1.")
    counts = sub["image_id"].value_counts()
    if len(counts) and counts.max() > MAX_BOXES_PER_IMAGE:
        fail(f"ảnh {counts.idxmax()} có {int(counts.max())} hộp; tối đa {MAX_BOXES_PER_IMAGE} hộp mỗi ảnh.")
    boxes = num[:, :4].copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
    img = sub["image_id"].to_numpy(object)
    ap50 = average_precision(gt, img, boxes, num[:, 4], 0.5)
    ap75 = average_precision(gt, img, boxes, num[:, 4], 0.75)
    n_gt = int(sum(int((~ig).sum()) for _, ig in gt.values()))
    metrics = {"AP50": ap50, "AP75": ap75, "n_gt": n_gt, "n_pred": len(sub)}
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(ap50))


if __name__ == "__main__":
    main()
