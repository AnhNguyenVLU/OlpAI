from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

THRESHOLD = Threshold(baseline=0.728, target=0.925)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên
EDGE_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


def syllables(text: str) -> list[str]:
    out = []
    for tok in unicodedata.normalize("NFC", str(text)).lower().split():
        tok = EDGE_RE.sub("", tok)
        if any(c.isalpha() for c in tok):
            out.append(tok)
    return out


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Chấm bài A — Phục hồi dấu tiếng Việt (độ chính xác âm tiết, trung bình theo câu).")
    ap.add_argument("--pred", type=Path, default=Path("submission.csv"))
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gold_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gold_dir / "answers.csv", args.split)
    gold = pd.read_csv(gold_dir / "answers.csv", dtype=str, keep_default_na=False)
    pred = read_submission(args.pred, ["id", "text"], dtype=str, keep_default_na=False)
    pred["id"] = pred["id"].str.strip()
    if (pred["id"] == "").any():
        fail("có id rỗng.")
    dup = pred["id"][pred["id"].duplicated()]
    if len(dup):
        fail(f"id bị trùng, ví dụ {dup.iloc[0]}.")
    missing = set(gold["id"]) - set(pred["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    extra = set(pred["id"]) - set(gold["id"])
    if extra:
        fail(f"có {len(extra)} id không thuộc tập test, ví dụ {sorted(extra)[0]}.")
    if (pred["text"].str.strip() == "").any():
        fail("có dòng text rỗng.")

    p = dict(zip(pred["id"], pred["text"]))
    per_sent, n_ok, n_tot, n_sent_ok = [], 0, 0, 0
    for i, g in zip(gold["id"], gold["text"]):
        gs, ps = syllables(g), syllables(p[i])
        if not gs:  # câu đáp án không có âm tiết: đúng khi dự đoán cũng không có
            per_sent.append(float(not ps))
            n_sent_ok += int(not ps)
            continue
        ok = sum(a == b for a, b in zip(gs, ps)) if len(gs) == len(ps) else 0
        n_ok += ok
        n_tot += len(gs)
        per_sent.append(ok / len(gs))
        n_sent_ok += int(ok == len(gs))
    arr = np.array(per_sent)
    score = float(arr.mean())
    metrics = {
        "syllable_acc_mean": score,
        "syllable_acc_micro": n_ok / max(n_tot, 1),
        "sentence_acc": n_sent_ok / len(arr),
        "ci95": float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))),
    }
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(score))


if __name__ == "__main__":
    main()
