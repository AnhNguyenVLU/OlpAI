from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

DR, DC = [-1, 0, 1, 0], [0, 1, 0, -1]  # hướng 0 lên, 1 phải, 2 xuống, 3 trái


def state_features(df: pd.DataFrame) -> pd.DataFrame:
    # vài đặc trưng thô: robot, có đang mang vật không, loại + màu ô ngay trước mặt robot
    rows = []
    for huong, mang, loai, mau, robot in zip(df["huong"], df["mang_loai"], df["luoi_loai"], df["luoi_mau"], df["robot"]):
        pos = loai.index("8")
        r, c = divmod(pos, 9)
        f = (r + DR[huong]) * 9 + (c + DC[huong])
        rows.append((str(robot), str(int(mang > 0)), loai[f], mau[f]))
    return pd.DataFrame(rows, columns=["robot", "mang", "truoc_loai", "truoc_mau"])


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Baseline Bài B: LogisticRegression trên vài đặc trưng thô của trạng thái "
                                             "(robot, đang mang vật, ô phía trước) và túi từ của nhiệm vụ.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "observations.csv")
    require_data(test_dir / "observations.csv", args.split)
    train = pd.read_csv(args.data / "train" / "observations.csv", dtype={"luoi_loai": str, "luoi_mau": str})
    test = pd.read_csv(test_dir / "observations.csv", dtype={"luoi_loai": str, "luoi_mau": str})

    enc = OneHotEncoder(handle_unknown="ignore")
    vec = CountVectorizer(binary=True)
    X_tr = sparse.hstack([enc.fit_transform(state_features(train)), vec.fit_transform(train["nhiem_vu"])]).tocsr()
    X_te = sparse.hstack([enc.transform(state_features(test)), vec.transform(test["nhiem_vu"])]).tocsr()
    model = LogisticRegression(max_iter=2000)
    model.fit(X_tr, train["hanh_dong"])
    pred = model.predict(X_te)
    pd.DataFrame({"id": test["id"], "hanh_dong": np.asarray(pred, int)}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
