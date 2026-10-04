"""Tiện ích chấm điểm dùng chung cho mọi bài.

Quy đổi metric thô sang thang 0–100 theo kiểu IOAI/OlpAI:

    Điểm = 100 · clip((metric − baseline) / (target − baseline), 0, 1)

- ``baseline``: điểm thô của lời giải cơ sở (baseline.py) → 0 điểm.
- ``target``: điểm thô "Điểm BTC" → 100 điểm.
Hỗ trợ cả metric càng lớn càng tốt và càng nhỏ càng tốt (target < baseline).
"""

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
    """Tên task chuẩn: ``<thư mục đề>/<thư mục bài>[<split>]``, lấy từ vị trí score.py."""
    task_dir = Path(script).resolve().parent
    return f"{task_dir.parent.name}/{task_dir.name}[{split}]"


def fail(message: str) -> NoReturn:
    """Báo lỗi bài nộp/dữ liệu và thoát với mã 1."""
    sys.stderr.write(f"LỖI: {message}\n")
    raise SystemExit(1)


def require_data(path: Path, split: str = "public") -> None:
    """Thoát với hướng dẫn rõ ràng nếu thư mục dữ liệu chưa tồn tại."""
    if path.exists():
        return
    if split == "public":
        fail(f"không tìm thấy {path}. Hãy giải nén dữ liệu trước: `unzip public_data.zip` tại thư mục bài.")
    fail(f"không tìm thấy {path}. Tập chấm ẩn chỉ có ở máy giáo viên/BTC.")


def read_submission(path: str | Path, required: Iterable[str], **read_csv_kwargs) -> pd.DataFrame:
    """Đọc file nộp CSV (UTF-8, chấp nhận BOM/CRLF), chuẩn hoá tên cột và kiểm tra cột bắt buộc.

    Tên cột được so khớp không phân biệt hoa thường/khoảng trắng; cột thừa bị bỏ qua.
    Trả về DataFrame chỉ gồm các cột bắt buộc theo đúng thứ tự ``required``.
    """
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
    """In kết quả chấm dưới dạng JSON một dòng (dễ parse trong CI)."""
    payload = {"task": task, "metrics": {k: round(float(v), 6) for k, v in metrics.items()}, "points": points}
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
