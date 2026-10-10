from __future__ import annotations

# Suy luận numpy cho hai "giám định viên" của Bài A (score.py dùng chính các hàm này để chấm).
#   M = mạng MLP một lớp ẩn ReLU trên ảnh đã chuẩn hoá theo từng pixel.
#   F = rừng cây quyết định (bỏ phiếu mềm), mỗi nút so sánh một pixel với một ngưỡng.
# Ảnh: mảng float32 (N, 32, 32, 3) giá trị trong [0, 1]; pixel thứ j của ảnh làm phẳng là x.reshape(N, -1)[:, j]
# (thứ tự hàng → cột → kênh R, G, B).

from pathlib import Path

import numpy as np

EPS = 16 / 255  # ràng buộc |delta| <= EPS ở mọi phần tử
EPS_TOL = 1e-4  # dung sai khi kiểm tra ràng buộc (sai số làm tròn float16)
H = W = 32
D = H * W * 3
CLASSES = [
    "cam_nguoc_chieu",
    "cam_re_trai",
    "toc_do_40",
    "toc_do_50",
    "toc_do_60",
    "duong_uu_tien",
    "nguoi_di_bo",
    "tre_em",
    "cong_truong",
    "dung_lai",
    "cam_do_xe",
    "cam_dung_do_xe",
]


def load_models(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def to_float(images_u8: np.ndarray) -> np.ndarray:
    return np.asarray(images_u8, np.float32) / np.float32(255)


def apply_delta(x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    # Đúng quy trình lúc chấm: delta ép về float32, cộng vào ảnh float32 rồi cắt về [0, 1].
    return np.clip(np.asarray(x, np.float32) + np.asarray(delta).astype(np.float32), 0.0, 1.0)


def mlp_logits(m: dict, x: np.ndarray) -> np.ndarray:
    X = np.asarray(x, np.float32).reshape(len(x), -1).astype(np.float64)
    z = (X - m["mlp_mean"]) / m["mlp_scale"]
    h = np.maximum(z @ m["mlp_W1"] + m["mlp_b1"], 0.0)
    return h @ m["mlp_W2"] + m["mlp_b2"]


def forest_leaves(m: dict, x: np.ndarray) -> np.ndarray:
    # Chỉ số nút lá (toàn cục) mà mỗi ảnh rơi vào ở từng cây: mảng (N, số cây).
    # Tại nút trong k: đi sang trái nếu x[feature[k]] <= threshold[k] (so sánh float32), ngược lại sang phải.
    X = np.asarray(x, np.float32).reshape(len(x), -1)
    feat, thr = m["forest_feature"], m["forest_threshold"]
    left, right = m["forest_left"], m["forest_right"]
    node = np.repeat(m["forest_roots"][None, :], X.shape[0], axis=0)
    rows = np.arange(X.shape[0])[:, None]
    for _ in range(int(m["forest_max_depth"])):
        f = feat[node]
        inner = f >= 0
        if not inner.any():
            break
        go_left = X[rows, np.where(inner, f, 0)] <= thr[node]
        node = np.where(inner, np.where(go_left, left[node], right[node]), node)
    return node


def forest_proba(m: dict, x: np.ndarray) -> np.ndarray:
    # Trung bình phân phối lớp tại lá của các cây (bỏ phiếu mềm).
    return m["forest_value"][forest_leaves(m, x)].astype(np.float64).mean(axis=1)


def is_correct(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    # "Đúng" nghĩa là điểm của lớp thật lớn hơn HẲN mọi lớp khác (hoà = sai).
    scores = np.asarray(scores, np.float64)
    y = np.asarray(y, np.int64)
    rows = np.arange(len(y))
    true = scores[rows, y]
    other = scores.copy()
    other[rows, y] = -np.inf
    return true > other.max(axis=1)


def predict(m: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return mlp_logits(m, x).argmax(axis=1), forest_proba(m, x).argmax(axis=1)


def penalty_factor(delta: np.ndarray) -> np.ndarray:
    # PF = 1 - 0.5 * ||delta||_2 / L2_MAX, với L2_MAX = EPS * sqrt(32*32*3): nhiễu dùng hết ngân sách → PF = 0.5.
    d = np.asarray(delta).astype(np.float64).reshape(len(delta), -1)
    l2 = np.sqrt((d * d).sum(axis=1))
    return 1.0 - 0.5 * np.minimum(1.0, l2 / (EPS * np.sqrt(D)))


def evaluate(m: dict, x: np.ndarray, y: np.ndarray, delta_a: np.ndarray, delta_b: np.ndarray) -> dict[str, np.ndarray]:
    # Kết quả từng ảnh đúng như score.py: thành công loại A/B và PF tương ứng.
    xa, xb = apply_delta(x, delta_a), apply_delta(x, delta_b)
    ok_a = ~is_correct(mlp_logits(m, xa), y) & is_correct(forest_proba(m, xa), y)
    ok_b = is_correct(mlp_logits(m, xb), y) & ~is_correct(forest_proba(m, xb), y)
    return {"success_a": ok_a, "success_b": ok_b, "pf_a": penalty_factor(delta_a), "pf_b": penalty_factor(delta_b)}
