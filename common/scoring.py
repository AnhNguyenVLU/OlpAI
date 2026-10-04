from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NoReturn

import pandas as pd


@dataclass(frozen=True)
class Threshold:
    baseline: float
    target: float

    def to_points(self, metric: float) -> float:
        span = self.target - self.baseline
        if span == 0:
            return 100.0 if metric == self.target else 0.0
        ratio = (metric - self.baseline) / span
        return round(100.0 * min(max(ratio, 0.0), 1.0), 2)


def task_name(script: str | Path, split: str) -> str:
    task_dir = Path(script).resolve().parent
    return f"{task_dir.parent.name}/{task_dir.name}[{split}]"


def fail(message: str) -> NoReturn:
    sys.stderr.write(f"LỖI: {message}\n")
    raise SystemExit(1)


def require_data(path: Path, split: str = "public") -> None:
    if path.exists():
        return
    if split == "public":
        fail(f"không tìm thấy {path}. Hãy giải nén dữ liệu trước: `unzip public_data.zip` tại thư mục bài.")
    fail(f"không tìm thấy {path}. Tập chấm ẩn chỉ có ở máy giáo viên/BTC.")


def read_submission(path: str | Path, required: Iterable[str], **read_csv_kwargs) -> pd.DataFrame:
    required = list(required)
    path = Path(path)
    if not path.exists():
        fail(f"không tìm thấy file nộp {path}.")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", **read_csv_kwargs)
    except Exception as exc:  # noqa: BLE001 - mọi lỗi parse đều là lỗi định dạng
        fail(f"không đọc được {path} dưới dạng CSV UTF-8 ({exc}).")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        hint = ""
        if not any(c in df.columns for c in required):
            hint = " Có thể file thiếu dòng header — dòng đầu tiên phải là tên cột."
        fail(f"thiếu cột {missing}; cần {required}, nhận được {list(df.columns)}.{hint}")
    return df[required]


def report(task: str, metrics: dict[str, float], points: float) -> None:
    payload = {"task": task, "metrics": {k: round(float(v), 6) for k, v in metrics.items()}, "points": points}
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
