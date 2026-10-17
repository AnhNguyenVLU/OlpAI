from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import Threshold, fail, read_submission, report, require_data, task_name  # noqa: E402

ASPECTS = ["vi_tri", "phong", "ve_sinh", "phuc_vu", "gia_ca", "an_uong", "tien_ich", "yen_tinh"]
POLARITIES = ["pos", "neg"]
THRESHOLD = Threshold(baseline=0.725, target=0.97)  # đo trên tập ẩn, chi tiết trong bộ công cụ giáo viên


def parse(cell: str, row_id: str, strict: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in str(cell).split():
        asp, sep, pol = tok.partition(":")
        if strict and (not sep or asp not in ASPECTS or pol not in POLARITIES):
            fail(f"id {row_id}: nhãn không hợp lệ '{tok}' (cần dạng khia_canh:pos hoặc khia_canh:neg).")
        if strict and asp in out:
            fail(f"id {row_id}: khía cạnh '{asp}' xuất hiện hơn một lần.")
        out[asp] = pol
    return out


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0


def counts(truth: list[dict[str, str]], pred: list[dict[str, str]]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for t, p in zip(truth, pred):
        ts, ps = set(t.items()), set(p.items())
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    return tp, fp, fn


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Chấm Bài A: micro-F1 trên các bộ ba (đánh giá, khía cạnh, cực tính).")
    ap.add_argument("--pred", type=Path, default=here / "submission.csv")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    gt_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(gt_dir / "answers.csv", args.split)
    gt = pd.read_csv(gt_dir / "answers.csv", dtype=str, keep_default_na=False)

    sub = read_submission(args.pred, ["id", "labels"], dtype=str, keep_default_na=False)
    sub["id"] = sub["id"].str.strip()
    if (sub["id"] == "").any():
        fail("có id rỗng.")
    if sub["id"].duplicated().any():
        fail(f"trùng id: {sub.loc[sub['id'].duplicated(), 'id'].iloc[0]}.")
    missing = set(gt["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(gt["id"])
    if missing:
        fail(f"thiếu {len(missing)} id, ví dụ {sorted(missing)[0]}.")
    if extra:
        fail(f"có {len(extra)} id lạ, ví dụ {sorted(extra)[0]}.")
    pred_map = {i: parse(c, i, True) for i, c in zip(sub["id"], sub["labels"])}

    truth = [parse(c, i, False) for i, c in zip(gt["id"], gt["labels"])]
    pred = [pred_map[i] for i in gt["id"]]
    tp, fp, fn = counts(truth, pred)
    micro = f1(tp, fp, fn)
    # phát hiện khía cạnh (bỏ cực tính) và độ chính xác cực tính trên các khía cạnh phát hiện đúng
    a_tp = sum(len(set(t) & set(p)) for t, p in zip(truth, pred))
    a_fp = sum(len(set(p) - set(t)) for t, p in zip(truth, pred))
    a_fn = sum(len(set(t) - set(p)) for t, p in zip(truth, pred))
    pol_ok = sum(sum(t[a] == p[a] for a in set(t) & set(p)) for t, p in zip(truth, pred))
    metrics = {"micro_f1": micro, "aspect_f1": f1(a_tp, a_fp, a_fn),
               "polarity_acc": pol_ok / a_tp if a_tp else 0.0}
    for col, name in (("khong_dau", "f1_khong_dau"), ("hon_hop", "f1_hon_hop")):
        if col in gt.columns:
            m = (gt[col] == "1").to_numpy()
            metrics[name] = f1(*counts([t for t, k in zip(truth, m) if k], [p for p, k in zip(pred, m) if k]))
    metrics["n_reviews"] = float(len(gt))
    report(task_name(__file__, args.split), metrics, THRESHOLD.to_points(micro))


if __name__ == "__main__":
    main()
