from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài B: LogisticRegression trên các đặc trưng số (impute median + chuẩn hoá).")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "train.csv")
    require_data(test_dir / "test.csv", args.split)
    train = pd.read_csv(args.data / "train" / "train.csv")
    test = pd.read_csv(test_dir / "test.csv")

    num_cols = [c for c in train.select_dtypes("number").columns if c not in ("vo_no",)]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[num_cols], train["vo_no"])
    prob = model.predict_proba(test[num_cols])[:, 1]
    pd.DataFrame({"id": test["id"], "prob": prob.round(6)}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
