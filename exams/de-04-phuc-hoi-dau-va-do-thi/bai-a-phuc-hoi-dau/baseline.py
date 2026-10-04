from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

EDGE_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s.replace("đ", "d").replace("Đ", "D"))


def split_token(tok: str) -> tuple[str, str, str]:
    m = EDGE_RE.match(tok)
    return m.group(1), m.group(2), m.group(3)


def apply_case(src: str, dst: str) -> str:
    if len(src) != len(dst):
        return dst
    return "".join(d.upper() if s.isupper() else d for s, d in zip(src, dst))


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline bài A — unigram: mỗi âm tiết không dấu -> dạng có dấu phổ biến nhất trong train.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train", "public")
    require_data(test_dir, args.split)

    train = pd.read_csv(args.data / "train" / "data.csv")
    counts: dict[str, Counter] = defaultdict(Counter)
    for text in train["text"]:
        for tok in text.split():
            word = split_token(tok)[1].lower()
            if word:
                counts[strip_accents(word)][word] += 1
    best = {k: c.most_common(1)[0][0] for k, c in counts.items()}

    test = pd.read_csv(test_dir / "data.csv", keep_default_na=False)
    out = []
    for text in test["text"]:
        toks = []
        for tok in text.split():
            pre, word, post = split_token(tok)
            rep = best.get(word.lower())
            toks.append(pre + (apply_case(word, rep) if rep else word) + post)
        out.append(" ".join(toks))
    pd.DataFrame({"id": test["id"], "text": out}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
