from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.scoring import require_data  # noqa: E402

N_BANDS = 16


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    if x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    return sr, x


def clip_features(path: Path) -> np.ndarray:
    # Thống kê phổ toàn clip: trung bình và độ lệch chuẩn log-năng lượng của 16 dải tần (chia đều theo log).
    sr, x = read_wav(path)
    f, _, z = stft(x, fs=sr, nperseg=256, noverlap=128)
    power = np.abs(z) ** 2
    edges = np.geomspace(150, sr / 2, N_BANDS + 1)
    bands = np.stack([power[(f >= lo) & (f < hi)].sum(axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    logb = np.log(bands + 1e-8)
    return np.concatenate([logb.mean(axis=1), logb.std(axis=1)])


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Baseline Bài A: thống kê phổ toàn clip (16 dải) + One-vs-Rest LogisticRegression.")
    ap.add_argument("--data", type=Path, default=here / "data")
    ap.add_argument("--out", type=Path, default=here / "submission.csv")
    ap.add_argument("--split", choices=["public", "private"], default="public")
    args = ap.parse_args()

    test_dir = args.data / ("test_public" if args.split == "public" else "_private/test")
    require_data(args.data / "train" / "labels.csv")
    require_data(test_dir / "audio", args.split)
    species = pd.read_csv(args.data / "train" / "species.csv")["code"].tolist()
    train = pd.read_csv(args.data / "train" / "labels.csv")

    x_train = np.stack([clip_features(args.data / "train" / "audio" / f"{i}.wav") for i in train["id"]])
    y_train = np.stack([(train["species"] == c).to_numpy().astype(int) for c in species], axis=1)
    test_files = sorted((test_dir / "audio").glob("*.wav"))
    x_test = np.stack([clip_features(p) for p in test_files])

    model = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=2000)))
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)

    sub = pd.DataFrame(prob.round(5), columns=species)
    sub.insert(0, "id", [p.stem for p in test_files])
    sub.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
