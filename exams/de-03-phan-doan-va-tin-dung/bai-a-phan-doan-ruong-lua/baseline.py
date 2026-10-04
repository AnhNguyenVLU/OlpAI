"""Baseline Bài A: phân loại từng pixel theo màu RGB bằng RandomForest + đếm thành phần liên thông.

- Lấy mẫu ngẫu nhiên pixel từ ảnh train, đặc trưng = (R, G, B) thô.
- Dự đoán nhãn cho mọi pixel ảnh test.
- Đếm thửa: mask theo màu quá nhiễu để đếm thành phần liên thông (hàm count_fields bên dưới
  cho biết định nghĩa), nên baseline nộp hằng số = trung vị số thửa trong train.

Dùng: python baseline.py [--data ./data] [--out ./submission.csv] [--split public|private]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

MIN_FIELD_PIXELS = 12
PIXELS_PER_IMAGE = 100


def rle_encode(mask: np.ndarray) -> str:
    """Mask nhãn (flatten row-major) -> 'nhan:do_dai nhan:do_dai ...'."""
    flat = mask.ravel()
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    lengths = np.diff(np.r_[starts, flat.size])
    return " ".join(f"{int(flat[s])}:{int(n)}" for s, n in zip(starts, lengths))


def count_fields(mask: np.ndarray) -> int:
    """Số thành phần liên thông 4-láng giềng của lớp 0 có >= 12 pixel (định nghĩa 'thửa ruộng')."""
    lab, n = ndimage.label(mask == 0)
    if n == 0:
        return 0
    return int((np.bincount(lab.ravel())[1:] >= MIN_FIELD_PIXELS).sum())


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "answers.csv")
    require_data(test_dir / "images", args.split)
    train = pd.read_csv(args.data / "train" / "answers.csv")
    median_count = int(train["count"].median())
    xs, ys = [], []
    for iid in train["id"]:
        img = np.asarray(Image.open(args.data / "train" / "images" / f"{iid}.png")).reshape(-1, 3)
        msk = np.asarray(Image.open(args.data / "train" / "masks" / f"{iid}.png")).ravel()
        idx = rng.choice(msk.size, PIXELS_PER_IMAGE, replace=False)
        xs.append(img[idx])
        ys.append(msk[idx])
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=args.seed)
    clf.fit(np.concatenate(xs), np.concatenate(ys))

    rows = []
    for path in sorted((test_dir / "images").glob("*.png")):
        iid = path.stem
        img = np.asarray(Image.open(path).convert("RGB"))
        pred = clf.predict(img.reshape(-1, 3)).reshape(img.shape[:2]).astype(np.uint8)
        rows.append({"id": iid, "mask": rle_encode(pred), "count": median_count})
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
