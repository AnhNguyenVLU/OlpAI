from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def color_hist(path: Path) -> np.ndarray:
    # chia ảnh cho màu trung vị (≈ màu nền) để bớt ám màu, giữ các điểm ảnh lệch xa nền, histogram 4x4x4
    im = np.asarray(Image.open(path).convert("RGB"), np.float32)
    im = np.clip(im / (np.median(im.reshape(-1, 3), 0) + 1.0) * 160.0, 0, 255)
    px = im[np.abs(im - 160.0).max(-1) > 40]
    q = px.astype(np.int32) // 64
    return np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64) / 4096.0


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Baseline Bài A: mỗi từ có một 'mẫu histogram màu' học từ train; điểm ảnh = "
                                             "tổng tích vô hướng giữa histogram màu của ảnh và mẫu của các từ trong chú thích "
                                             "(đổi dấu ở vế câu có 'không').")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "captions.csv")
    require_data(test_dir / "queries.csv", args.split)
    train = pd.read_csv(args.data / "train" / "captions.csv", dtype=str)
    hist = {i: color_hist(args.data / "train" / "images" / f"{i}.png") for i in train["image_id"].unique()}

    # mẫu của từ w = histogram trung bình của các ảnh có chú thích chứa w − histogram trung bình chung
    H = np.stack([hist[i] for i in train["image_id"]])
    mean_all = H.mean(0)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for h, cap in zip(H, train["caption"]):
        for w in words(cap):
            sums[w] = sums.get(w, 0) + h
            counts[w] = counts.get(w, 0) + 1
    template = {w: sums[w] / counts[w] - mean_all for w in sums if counts[w] >= 5}

    queries = pd.read_csv(test_dir / "queries.csv", dtype=str)
    rows = []
    for qid, cap, cand in zip(queries["query_id"], queries["caption"], queries["candidates"]):
        t = np.zeros(64)
        for part in re.split(r"[,;.]", cap):
            sign = -1.0 if "không" in words(part) else 1.0
            t += sign * sum((template[w] for w in words(part) if w in template), np.zeros(64))
        ids = cand.split()
        score = np.array([color_hist(test_dir / "images" / f"{i}.png") @ t for i in ids])
        order = np.argsort(-score, kind="stable")
        rows.append((qid, " ".join(ids[k] for k in order)))
    pd.DataFrame(rows, columns=["query_id", "ranking"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
