from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as MD  # noqa: E402

TRIES = 10


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài A: nhiễu dấu ngẫu nhiên ở biên epsilon, thử vài lần và giữ nhiễu đầu tiên đạt yêu cầu.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.npz")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "models.npz")
    require_data(test_dir / "labels.csv", args.split)
    m = MD.load_models(args.data / "models.npz")
    lab = pd.read_csv(test_dir / "labels.csv", dtype={"id": str})
    ids = lab["id"].to_numpy()
    y = lab["label"].to_numpy(np.int64)
    x = MD.to_float(np.stack([np.asarray(Image.open(test_dir / "images" / f"{i}.png").convert("RGB")) for i in ids]))

    delta_a = np.zeros(x.shape, np.float16)
    delta_b = np.zeros(x.shape, np.float16)
    done_a = np.zeros(len(ids), bool)
    done_b = np.zeros(len(ids), bool)
    for _ in range(TRIES):
        noise = (MD.EPS * rng.choice([-1.0, 1.0], size=x.shape)).astype(np.float16)
        r = MD.evaluate(m, x, y, noise, noise)
        new_a = r["success_a"] & ~done_a
        new_b = r["success_b"] & ~done_b
        delta_a[new_a] = noise[new_a]
        delta_b[new_b] = noise[new_b]
        done_a |= new_a
        done_b |= new_b

    # ghi đúng vào đường dẫn --out (np.savez tự thêm đuôi .npz nếu truyền tên file dạng chuỗi)
    with open(args.out, "wb") as fh:
        np.savez_compressed(fh, ids=ids.astype(str), delta_a=delta_a, delta_b=delta_b)


if __name__ == "__main__":
    main()
