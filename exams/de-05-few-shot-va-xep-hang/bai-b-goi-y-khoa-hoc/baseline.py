"""Baseline bài B: gợi ý top-10 khoá phổ biến nhất toàn cục (theo tổng lượt ghi danh trong train).

Với mỗi người dùng cần gợi ý: lấy danh sách khoá xếp theo tổng lượt ghi danh của nhóm người học chính
(train/interactions.csv), bỏ các khoá người đó đã ghi danh (theo lịch sử trong thư mục test), giữ 10 khoá đầu.

Dùng: python baseline.py [--data ./data] [--split public|private] [--out ./submission.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

HERE = Path(__file__).resolve().parent
K = 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "interactions.csv", "public")
    require_data(test_dir / "test_users.csv", args.split)
    train = pd.read_csv(args.data / "train" / "interactions.csv")
    hist = pd.read_csv(test_dir / "interactions.csv")
    test_users = pd.read_csv(test_dir / "test_users.csv")["user_id"]

    ranking = train.loc[train["event"] == "enroll", "course_id"].value_counts().index.tolist()
    seen = hist[hist["event"] == "enroll"].groupby("user_id")["course_id"].agg(set).to_dict()

    rows = []
    for u in test_users:
        s = seen.get(u, set())
        rec = [c for c in ranking[: K + len(s)] if c not in s][:K]
        rows.append((u, " ".join(rec)))
    pd.DataFrame(rows, columns=["user_id", "course_ids"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
