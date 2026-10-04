"""Baseline bài A: dịch từng từ bằng từ điển đồng xuất hiện (hệ số Dice), giữ nguyên trật tự nguồn.

- Mỗi từ nguồn (nguyên dạng bề mặt, không tách hậu tố) -> các âm tiết tiếng Việt có Dice cao nhất.
- Từ chưa gặp: viết hoa thì chép nguyên (tên riêng), ngược lại bỏ qua.
Dùng: python baseline.py --data data --out submission.csv [--split public|private]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def build_dictionary(src: list[str], tgt: list[str], ratio: float = 0.9) -> dict[str, list[str]]:
    src_cnt: Counter = Counter()
    tgt_cnt: Counter = Counter()
    pair_cnt: dict[str, Counter] = defaultdict(Counter)
    pos_sum: dict[str, float] = defaultdict(float)
    for s, t in zip(src, tgt):
        stoks, ttoks = set(s.split()), t.split()
        for i, w in enumerate(ttoks):
            pos_sum[w] += i / max(len(ttoks) - 1, 1)
        tset = set(ttoks)
        src_cnt.update(stoks)
        tgt_cnt.update(tset)
        for a in stoks:
            pair_cnt[a].update(tset)
    tgt_freq = Counter(w for t in tgt for w in t.split())
    avg_pos = {w: pos_sum[w] / tgt_freq[w] for w in tgt_freq}
    lex: dict[str, list[str]] = {}
    for a, cnts in pair_cnt.items():
        dice = {b: 2 * c / (src_cnt[a] + tgt_cnt[b]) for b, c in cnts.items()}
        best = max(dice.values())
        chosen = [b for b, d in dice.items() if d >= ratio * best]
        lex[a] = sorted(chosen, key=lambda b: avg_pos[b])[:3]
    return lex


def translate(sent: str, lex: dict[str, list[str]]) -> str:
    out: list[str] = []
    for w in sent.split():
        if w in lex:
            out.extend(lex[w])
        elif w[:1].isupper():
            out.append(w)
    return " ".join(out) if out else "?"


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train", "public")
    require_data(test_dir, args.split)
    tr = pd.read_csv(args.data / "train" / "data.csv", dtype=str, keep_default_na=False)
    tr_ans = pd.read_csv(args.data / "train" / "answers.csv", dtype=str, keep_default_na=False)
    tr = tr.merge(tr_ans, on="id")
    te = pd.read_csv(test_dir / "data.csv", dtype=str, keep_default_na=False)

    lex = build_dictionary(tr["src"].tolist(), tr["translation"].tolist())
    sub = pd.DataFrame({"id": te["id"], "translation": [translate(s, lex) for s in te["src"]]})
    sub.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
