from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

W_AUC, W_PT, RADIUS = 0.7, 0.3, 3
THRESHOLD = Threshold(baseline=0.413, target=0.795)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def dilate(masks: np.ndarray, r: int) -> np.ndarray:
    out = masks.copy()
    n, h, w = masks.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                continue
            sh = np.zeros_like(masks)
            sh[:, max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
                masks[:, max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            out |= sh
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Chấm Bài B: 0.7·AUROC (phát hiện ảnh lỗi) + 0.3·Pointing (chỉ đúng vị trí lỗi).")
    ap.add_argument("--pred", type=Path, default=Path("submission.csv"))
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    require_data(gt_dir / "answers_masks.npz", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype={"id": str}, keep_default_na=False)
    with np.load(gt_dir / "answers_masks.npz", allow_pickle=False) as z:
        mask_ids = [str(i) for i in z["ids"]]
        masks = z["masks"] > 0
    h, w = masks.shape[1:]

    sub = read_submission(args.pred, ["id", "score", "x", "y"], dtype={"id": str})
    if sub["id"].isna().any():
        fail("có id rỗng.")
    sub["id"] = sub["id"].str.strip()
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}.")
    missing = set(gt["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(gt["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")
    vals = {}
    for col in ("score", "x", "y"):
        if pd.api.types.is_bool_dtype(sub[col]):
            fail(f"cột {col} phải là số, không phải True/False.")
        v = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(v).any() or not np.isfinite(v).all():
            fail(f"cột {col} có giá trị rỗng/NaN/inf hoặc không phải số.")
        vals[col] = v
    if ((vals["x"] < 0) | (vals["x"] >= w)).any():
        fail(f"x phải thoả 0 <= x < {w} (x là chỉ số cột).")
    if ((vals["y"] < 0) | (vals["y"] >= h)).any():
        fail(f"y phải thoả 0 <= y < {h} (y là chỉ số hàng).")

    pos = {i: k for k, i in enumerate(sub["id"])}
    order = np.array([pos[i] for i in gt["id"]])
    score, x, y = vals["score"][order], vals["x"][order], vals["y"][order]
    label = gt["co_loi"].astype(int).to_numpy()
    group = gt["nhom_loi"].astype(str).to_numpy()

    auc = float(roc_auc_score(label, score))
    mpos = {i: k for k, i in enumerate(mask_ids)}
    big = dilate(masks[[mpos[i] for i in gt["id"]]], RADIUS)
    rx = np.clip(np.floor(x + 0.5), 0, w - 1).astype(int)
    ry = np.clip(np.floor(y + 0.5), 0, h - 1).astype(int)
    hit = big[np.arange(len(gt)), ry, rx]
    pointing = float(hit[label == 1].mean())
    final = W_AUC * auc + W_PT * pointing

    metrics = {"diem_tho": final, "auroc": auc, "pointing": pointing}
    for g in ("da_biet", "chua_thay"):
        sel = (label == 0) | (group == g)
        if (group == g).any():
            metrics[f"auroc_{g}"] = float(roc_auc_score(label[sel], score[sel]))
            metrics[f"pointing_{g}"] = float(hit[group == g].mean())
    metrics["n_anh"] = float(len(gt))
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(final))


if __name__ == "__main__":
    main()
