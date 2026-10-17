from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Baseline Bài A: TF-IDF từ (1–2 gram) trên văn bản gốc + One-vs-Rest "
                                             "LogisticRegression cho 16 nhãn khia_canh:cuc_tinh, ngưỡng 0.5.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "train.csv")
    require_data(test_dir / "test.csv", args.split)
    train = pd.read_csv(args.data / "train" / "train.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(test_dir / "test.csv", dtype=str, keep_default_na=False)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train["labels"].str.split())
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000))
    clf.fit(vec.fit_transform(train["text"]), y)
    proba = clf.predict_proba(vec.transform(test["text"]))

    # mỗi khía cạnh: giữ cực tính có xác suất cao hơn nếu xác suất đó >= 0.5
    preds = []
    for row in proba:
        best: dict[str, tuple[float, str]] = {}
        for p, lab in zip(row, mlb.classes_):
            asp = lab.split(":")[0]
            if p >= 0.5 and p > best.get(asp, (0.0, ""))[0]:
                best[asp] = (float(p), lab)
        preds.append(" ".join(sorted(lab for _, lab in best.values())))
    pd.DataFrame({"id": test["id"], "labels": preds}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
