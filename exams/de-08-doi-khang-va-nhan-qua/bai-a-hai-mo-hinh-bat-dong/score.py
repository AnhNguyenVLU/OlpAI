from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, report, require_data, task_name  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as MD  # noqa: E402

THRESHOLD = Threshold(baseline=0.04, target=0.95)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def load_split(test_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    lab = pd.read_csv(test_dir / "labels.csv", dtype={"id": str})
    ids = lab["id"].tolist()
    images = np.stack([np.asarray(Image.open(test_dir / "images" / f"{i}.png").convert("RGB")) for i in ids])
    return ids, MD.to_float(images), lab["label"].to_numpy(np.int64)


def read_deltas(path: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        fail(f"không tìm thấy file nộp {path}.")
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as exc:  # noqa: BLE001 - mọi lỗi đọc đều là lỗi định dạng
        fail(f"không đọc được {path} dưới dạng .npz (np.savez/np.savez_compressed, không pickle): {exc}")
    for key in ("ids", "delta_a", "delta_b"):
        if key not in arrays:
            fail(f"thiếu mảng '{key}' trong file nộp; cần ids, delta_a, delta_b, nhận được {sorted(arrays)}.")
    sub_ids = arrays["ids"]
    if sub_ids.ndim != 1 or sub_ids.dtype.kind not in "US":
        fail("ids phải là mảng chuỗi một chiều (ví dụ np.array(['pub_000000', ...])).")
    try:
        sub_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in sub_ids.tolist()]
    except UnicodeDecodeError:
        fail("ids chứa byte không phải UTF-8.")
    if len(set(sub_ids)) != len(sub_ids):
        dup = pd.Series(sub_ids)
        fail(f"trùng id: {dup[dup.duplicated()].iloc[0]}.")
    missing = set(ids) - set(sub_ids)
    extra = set(sub_ids) - set(ids)
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")
    pos = {s: k for k, s in enumerate(sub_ids)}
    order = np.array([pos[i] for i in ids])
    out = []
    for key in ("delta_a", "delta_b"):
        d = arrays[key]
        if d.dtype not in (np.float16, np.float32):
            fail(f"{key} phải có kiểu float16 hoặc float32, nhận được {d.dtype}.")
        if d.shape != (len(sub_ids), MD.H, MD.W, 3):
            fail(f"{key} phải có kích thước ({len(sub_ids)}, {MD.H}, {MD.W}, 3), nhận được {d.shape}.")
        if not np.isfinite(d).all():
            fail(f"{key} chứa NaN hoặc vô cực.")
        worst = float(np.abs(d.astype(np.float64)).max(initial=0.0))
        if worst > MD.EPS + MD.EPS_TOL:
            fail(f"{key} vi phạm ràng buộc |delta| <= 16/255 (giá trị lớn nhất {worst:.5f} > {MD.EPS:.5f}).")
        out.append(d[order])
    return out[0], out[1]


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm Bài A: trung bình thành công x PF của nhiễu loại A (M sai, F đúng) và loại B (F sai, M đúng).")
    ap.add_argument("--pred", type=Path, default=here / "submission.npz")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "models.npz", args.split)
    require_data(test_dir / "labels.csv", args.split)
    m = MD.load_models(args.data / "models.npz")
    ids, x, y = load_split(test_dir)
    delta_a, delta_b = read_deltas(args.pred, ids)

    r = MD.evaluate(m, x, y, delta_a, delta_b)
    part_a = r["success_a"] * r["pf_a"]
    part_b = r["success_b"] * r["pf_b"]
    score = float((part_a.mean() + part_b.mean()) / 2)
    metrics = {
        "score": score,
        "success_a": float(r["success_a"].mean()),
        "success_b": float(r["success_b"].mean()),
        "pf_a_when_success": float(r["pf_a"][r["success_a"]].mean()) if r["success_a"].any() else 0.0,
        "pf_b_when_success": float(r["pf_b"][r["success_b"]].mean()) if r["success_b"].any() else 0.0,
    }
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(score))


if __name__ == "__main__":
    main()
