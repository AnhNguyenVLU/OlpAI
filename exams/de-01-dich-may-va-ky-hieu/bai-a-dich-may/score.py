"""Chấm bài A (dịch máy Kơ Ru -> Việt): chrF (β=1, n=1..6) toàn corpus, kèm BLEU-4 tham khảo.

Dùng: python score.py --pred submission.csv --data data [--split public|private]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

# Mốc điểm thô đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên.
BASELINE = 44.1
TARGET = 95.0

CHAR_ORDER = 6
BETA = 1.0  # β=1: precision và recall ngang nhau, không thưởng cho việc độn thêm từ


def char_ngrams(s: str, n: int) -> Counter:
    s = "".join(s.split())  # chrF bỏ khoảng trắng
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyps: list[str], refs: list[str]) -> float:
    """chrF toàn corpus (kiểu sacrebleu): cộng dồn thống kê n-gram trên mọi câu, trung bình P/R theo n rồi F_β."""
    match = [0] * CHAR_ORDER
    hyp_tot = [0] * CHAR_ORDER
    ref_tot = [0] * CHAR_ORDER
    for h, r in zip(hyps, refs):
        for n in range(1, CHAR_ORDER + 1):
            hc, rc = char_ngrams(h, n), char_ngrams(r, n)
            match[n - 1] += sum((hc & rc).values())
            hyp_tot[n - 1] += sum(hc.values())
            ref_tot[n - 1] += sum(rc.values())
    precs = [m / t if t else 0.0 for m, t in zip(match, hyp_tot)]
    recs = [m / t if t else 0.0 for m, t in zip(match, ref_tot)]
    p, r = sum(precs) / CHAR_ORDER, sum(recs) / CHAR_ORDER
    if p + r == 0:
        return 0.0
    b2 = BETA ** 2
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def bleu(hyps: list[str], refs: list[str], max_n: int = 4) -> float:
    """BLEU-4 toàn corpus trên token tách theo khoảng trắng (chỉ để tham khảo)."""
    match = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht, rt = h.split(), r.split()
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            hc = Counter(tuple(ht[i:i + n]) for i in range(len(ht) - n + 1))
            rc = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
            match[n - 1] += sum((hc & rc).values())
            total[n - 1] += max(len(ht) - n + 1, 0)
    if min(match) == 0 or hyp_len == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * bp * math.exp(log_p)


def load_pred(path: Path, ids: list[str]) -> list[str]:
    sub = read_submission(path, ["id", "translation"], dtype=str, keep_default_na=False, na_values=[])
    sub["id"] = sub["id"].str.strip()
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].head(5).tolist()}")
    missing = set(ids) - set(sub["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[:5]}")
    extra = set(sub["id"]) - set(ids)
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[:5]}")
    empty = sub["translation"].str.strip() == ""
    if empty.any():
        fail(f"{int(empty.sum())} bản dịch rỗng/NaN, ví dụ id {sub.loc[empty, 'id'].head(5).tolist()}")
    return sub.set_index("id").loc[ids, "translation"].tolist()


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    ans_path = args.data / ("test_public" if args.split == "public" else "_private/test") / "answers.csv"
    require_data(ans_path, args.split)
    ans = pd.read_csv(ans_path, dtype=str, keep_default_na=False)
    ids = ans["id"].tolist()
    hyps = load_pred(args.pred, ids)
    refs = ans["translation"].tolist()

    metrics = {"chrF": chrf(hyps, refs), "BLEU": bleu(hyps, refs)}
    if "subset" in ans.columns:
        for sub in sorted(ans["subset"].unique()):
            m = (ans["subset"] == sub).to_numpy()
            metrics[f"chrF_{sub}"] = chrf([h for h, k in zip(hyps, m) if k], [r for r, k in zip(refs, m) if k])
    points = Threshold(baseline=BASELINE, target=TARGET).to_points(metrics["chrF"])
    report(task_name(__file__, args.split), metrics, points)


if __name__ == "__main__":
    main()
