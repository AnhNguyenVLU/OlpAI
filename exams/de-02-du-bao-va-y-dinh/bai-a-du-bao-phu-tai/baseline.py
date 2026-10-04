"""Baseline Bài A — seasonal naive (cùng giờ tuần trước) với khoảng cố định ±15%.

- Lịch sử = train/load.csv + <split>/history.csv (số liệu đo sau train, có thể rỗng).
- P50(t) = load(t − 168h); nếu ô đó bị mất điện (0) hoặc thiếu số liệu thì dùng load(t − 336h);
  nếu vẫn không có (trạm mới < 2 tuần) thì dùng trung vị theo giờ-trong-ngày của lịch sử hợp lệ.
- P10 = 0.85 · P50, P90 = 1.15 · P50.

Cách dùng: python baseline.py [--data ./data] [--out ./submission.csv] [--split public|private]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    split_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train", "public")
    require_data(split_dir, args.split)
    parts = [pd.read_csv(f) for f in (args.data / "train" / "load.csv", split_dir / "history.csv")]
    hist = pd.concat([p for p in parts if len(p)], ignore_index=True)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist.loc[hist["load_mw"] <= 0, "load_mw"] = np.nan  # mất điện -> coi như thiếu
    hist = hist.dropna(subset=["load_mw"])
    test = pd.read_csv(split_dir / "test.csv", parse_dates=["timestamp"])

    sub = test.copy()
    for lag_h in (168, 336):
        lag = hist[["station_id", "timestamp", "load_mw"]].copy()
        lag["timestamp"] = lag["timestamp"] + pd.Timedelta(hours=lag_h)
        sub = sub.merge(lag.rename(columns={"load_mw": f"lag{lag_h}"}), on=["station_id", "timestamp"], how="left")

    hist["hour"] = hist["timestamp"].dt.hour
    hod = hist.groupby(["station_id", "hour"])["load_mw"].median().rename("hod").reset_index()
    sub["hour"] = sub["timestamp"].dt.hour
    sub = sub.merge(hod, on=["station_id", "hour"], how="left")
    p50 = sub["lag168"].fillna(sub["lag336"]).fillna(sub["hod"]).fillna(0.0)

    sub["p50"] = p50.round(4)
    sub["p10"] = (0.85 * p50).round(4)
    sub["p90"] = (1.15 * p50).round(4)
    sub["timestamp"] = sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    sub[["station_id", "timestamp", "p10", "p50", "p90"]].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
