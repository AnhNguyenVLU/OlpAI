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

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline Bài B — TF-IDF (từ) + One-vs-Rest Logistic Regression, ngưỡng 0.5.")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    split_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train", "public")
    require_data(split_dir, args.split)
    train = pd.read_csv(args.data / "train" / "data.csv").merge(
        pd.read_csv(args.data / "train" / "answers.csv", keep_default_na=False), on="id")
    test = pd.read_csv(split_dir / "data.csv")

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train["labels"].str.split("|"))
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_tr = vec.fit_transform(train["text"])
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=1000))
    clf.fit(x_tr, y)
    proba = clf.predict_proba(vec.transform(test["text"]))
    preds = ["|".join(mlb.classes_[i] for i in range(len(mlb.classes_)) if row[i] >= 0.5) for row in proba]
    pd.DataFrame({"id": test["id"], "labels": preds}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
