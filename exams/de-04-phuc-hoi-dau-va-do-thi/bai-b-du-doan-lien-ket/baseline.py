from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline bài B — Common Neighbors: điểm = số cộng sự chung trên đồ thị đồng tác giả đến 2023.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train", "public")
    require_data(test_dir, args.split)

    edges = pd.read_csv(args.data / "train" / "edges.csv")
    nbr: dict[str, set[str]] = defaultdict(set)
    for u, v in zip(edges["u"], edges["v"]):
        nbr[u].add(v)
        nbr[v].add(u)
    q = pd.read_csv(test_dir / "queries.csv")
    cand = pd.read_csv(test_dir / "candidates.csv").merge(q, on="query_id")
    cand["score"] = [len(nbr[a] & nbr[c]) for a, c in zip(cand["author_id"], cand["candidate_id"])]
    cand[["query_id", "candidate_id", "score"]].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
