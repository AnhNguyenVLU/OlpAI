from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

LABEL_COLS = ["nhan_voucher", "muc_giam", "mua_hang"]


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài B: mô hình đáp ứng P(mua_hang = 1 | đặc trưng), bỏ qua việc có nhận voucher hay không.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "train.csv")
    require_data(test_dir / "test.csv", args.split)
    train = pd.read_csv(args.data / "train" / "train.csv")
    test = pd.read_csv(test_dir / "test.csv")

    feats = [c for c in train.columns if c not in ["id"] + LABEL_COLS]
    both = pd.get_dummies(pd.concat([train[feats], test[feats]], ignore_index=True), dtype=float)
    X_train, X_test = both.iloc[: len(train)], both.iloc[len(train):]
    model = HistGradientBoostingClassifier(random_state=0).fit(X_train, train["mua_hang"])
    score = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"id": test["id"], "uplift": score.round(6)}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
