from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def relative_positions(turns: pd.DataFrame, answers: pd.DataFrame) -> np.ndarray:
    # vị trí thật / (n - 1) của từng dòng trong turns.csv (0 = lượt đầu, 1 = lượt cuối)
    order = {did: [int(t) for t in o.split()] for did, o in zip(answers["id"], answers["order"])}
    return np.array([order[d][k] / (len(order[d]) - 1) for d, k in zip(turns["id"], turns["turn"])])


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài B: TF-IDF từ đơn (500 từ phổ biến nhất) + Ridge hồi quy vị trí tương đối của từng lượt, rồi sắp xếp.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "turns.csv")
    require_data(test_dir / "turns.csv", args.split)
    tr_turns = pd.read_csv(args.data / "train" / "turns.csv", keep_default_na=False)
    tr_answers = pd.read_csv(args.data / "train" / "answers.csv", dtype={"order": str})
    te_turns = pd.read_csv(test_dir / "turns.csv", keep_default_na=False)
    prefix = pd.read_csv(test_dir / "prefix.csv").set_index("id")

    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_features=500, sublinear_tf=True)
    x_train = vec.fit_transform(tr_turns["text"])
    model = Ridge(alpha=1.0).fit(x_train, relative_positions(tr_turns, tr_answers))
    te_turns = te_turns.assign(score=model.predict(vec.transform(te_turns["text"])))

    rows = []
    for did, g in te_turns.groupby("id", sort=False):
        first, second = int(prefix.loc[did, "first"]), int(prefix.loc[did, "second"])
        score = dict(zip(g["turn"], g["score"]))
        rest = sorted((k for k in score if k not in (first, second)), key=lambda k: score[k])
        position = {first: 0, second: 1, **{k: p for p, k in enumerate(rest, start=2)}}
        rows.append((did, " ".join(str(position[k]) for k in range(len(score)))))
    pd.DataFrame(rows, columns=["id", "order"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
