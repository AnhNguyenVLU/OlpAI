from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

MIN_WORDS = 120
WORD = re.compile(r"[^\s,.!?:;]+")
# cụm từ "sáo rỗng" hay gặp trong văn bản do máy viết (người viết thỉnh thoảng cũng dùng)
CLICHES = [
    "bên cạnh đó", "ngoài ra", "có thể nói", "đặc biệt,", "không chỉ", "nhìn chung", "hơn nữa",
    "trên thực tế", "đáng chú ý", "góp phần", "đóng vai trò quan trọng", "mang lại nhiều giá trị",
]


def predict(text: str, frac: float) -> int:
    # cụm sáo rỗng đầu tiên xuất hiện sau 120 từ đầu; nếu không có: tỉ lệ ranh giới trung bình x độ dài
    words = list(WORD.finditer(text))
    start = words[MIN_WORDS - 1].end() if len(words) >= MIN_WORDS else 0
    low = text.lower()
    hits = [p for p in (low.find(c, start) for c in CLICHES) if p >= 0]
    return min(hits) if hits else int(round(frac * len(text)))


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài B: ranh giới = cụm từ sáo rỗng đầu tiên sau 120 từ đầu; không có thì lấy tỉ lệ ranh giới trung bình của train × độ dài.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "answers.csv")
    require_data(test_dir / "data.csv", args.split)
    train = pd.read_csv(args.data / "train" / "data.csv", dtype={"id": str}, keep_default_na=False)
    train = train.merge(pd.read_csv(args.data / "train" / "answers.csv", dtype={"id": str}), on="id")
    frac = float((train["boundary_char_index"] / train["text"].str.len()).mean())
    test = pd.read_csv(test_dir / "data.csv", dtype={"id": str}, keep_default_na=False)
    pred = [predict(t, frac) for t in test["text"]]
    pd.DataFrame({"id": test["id"], "boundary_char_index": pred}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
