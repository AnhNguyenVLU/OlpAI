# Bài B — Đôi tay biết nói

- **Thời gian chạy:** 10 phút (toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** CPU 4 nhân, 16 GB RAM (có thể có một GPU ≈16 GB), không có internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** macro-F1 = 0.583
- **Điểm BTC (100 điểm):** macro-F1 = 0.94

## Bối cảnh

Một trường chuyên biệt cho học sinh khiếm thính ở Đà Nẵng muốn làm ứng dụng giúp phụ huynh học **ngôn ngữ
ký hiệu**: phụ huynh ký trước camera điện thoại, ứng dụng cho biết họ vừa ký từ nào. Một bộ phát hiện bàn tay
đã chuyển mỗi video thành chuỗi **21 điểm mốc (keypoint) 2D của bàn tay** theo từng khung hình. Nhóm phát
triển đã thu được dữ liệu từ 24 giáo viên và học sinh. Nhưng ứng dụng sẽ được dùng bởi **những người hoàn toàn
mới**: tay to nhỏ khác nhau, cầm điện thoại nghiêng, ký nhanh chậm khác nhau, camera rung, có khung bị mất bàn
tay. Và không phải ai cũng **thuận tay phải**.

(Dữ liệu do BTC sinh tổng hợp từ một mô hình bàn tay 3D; 30 ký hiệu là các tổ hợp tư thế bàn tay và chuyển động.)

## Nhiệm vụ

Cho một chuỗi keypoints của một lần ký, dự đoán ký hiệu (1 trong 30 lớp).

## Dữ liệu

| Thư mục | Số mẫu | Người ký | `answers.csv`? | Dùng để |
|---|---|---|---|---|
| `data/train/` | 3 600 | 24 người (có cột `signer`) | ✅ có | huấn luyện |
| `data/test_public/` | 1 200 | 20 người **mới** | ✅ có (bản sao dev) | chạy pipeline và tự chấm cục bộ |
| tập chấm ẩn | 1 800 | 30 người mới khác | ❌ | chấm điểm chính thức |

- Người ký trong test **không xuất hiện trong train** (domain shift theo người ký). Tập chấm ẩn do BTC giữ,
  cùng định dạng và **cùng phân phối với `test_public`** (cùng kiểu người ký mới, nhưng là những người khác),
  không có đáp án; điểm `test_public` là ước lượng hợp lý.
- Các lớp cân bằng: ở train mỗi người ký thực hiện mỗi ký hiệu 5 lần, ở test 2 lần.
- Nhãn train có thể chứa **một ít nhiễu** (khoảng vài phần trăm bị gán nhầm).

```
data/train/data.npz          # X, lengths, ids, signer
data/train/answers.csv       # id,label
data/train/label_names.csv   # danh sách 30 nhãn hợp lệ
data/test_public/data.npz    # X, lengths, ids
data/test_public/answers.csv # id,label   (chỉ có ở bản sao dev)
```

Trong `data.npz`:

- `X`: `float16`, kích thước `(tổng số khung, 21, 2)` — toạ độ $(x, y)$ chuẩn hoá theo khung ảnh, $y$ hướng
  **xuống** (như MediaPipe). `NaN` = khung bị mất bàn tay hoặc điểm bị che.
- `lengths`: `int32 (N,)` — số khung $T_i$ của từng mẫu ($T$ thay đổi, khoảng 20–80).
- `ids`: mã mẫu. Mẫu $i$ là `X[o_i : o_i + lengths[i]]` với `o = cumsum(lengths) - lengths`.
- `signer` (chỉ train): mã người ký, dùng để chia validation theo người (GroupKFold).

Thứ tự 21 điểm: 0 cổ tay; 1–4 ngón cái; 5–8 ngón trỏ; 9–12 ngón giữa; 13–16 ngón áp út; 17–20 ngón út
(mỗi ngón từ gốc ra đầu ngón).

```python
import numpy as np
z = np.load("data/train/data.npz")
off = np.cumsum(z["lengths"]) - z["lengths"]
seqs = [z["X"][o:o + n].astype(np.float32) for o, n in zip(off, z["lengths"])]
```

Lấy dữ liệu: tại thư mục bài, chạy `unzip public_data.zip`.

## Định dạng nộp

File CSV hai cột `id,label`; `label` là một trong 30 nhãn trong `label_names.csv`:

```csv
id,label
pub_00000,me
pub_00001,di
```

Thiếu id, trùng id, id lạ, nhãn rỗng hoặc nhãn không có trong danh sách → bài nộp bị từ chối.

## Ràng buộc

- Chỉ dùng dữ liệu trong `data/train/`; không dữ liệu ngoài, không trọng số pretrained.
- Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn mô hình — chỉ dùng để tự chấm. Không dùng
  đặc trưng thống kê gom theo nhiều mẫu test (mỗi mẫu test được dự đoán độc lập).
- Tổng thời gian huấn luyện + suy luận lúc chấm ≤ 10 phút.

## Chấm điểm

Metric là **macro-F1** trên 30 lớp:

$$\text{F1}_c = \frac{2\,\text{TP}_c}{2\,\text{TP}_c + \text{FP}_c + \text{FN}_c},\qquad
\text{macro-F1} = \frac{1}{30}\sum_{c=1}^{30} \text{F1}_c$$

Mọi lớp có trọng số như nhau: một mô hình nhầm hệ thống một cặp ký hiệu (ví dụ hai ký hiệu chỉ khác nhau ở
chiều chuyển động) sẽ mất gần trọn F1 của cả hai lớp đó. `score.py` cũng báo accuracy để tham khảo.

$$\text{Điểm} = 100\cdot\operatorname{clip}\!\left(\frac{\text{macro-F1} - 0.583}{0.94 - 0.583},\ 0,\ 1\right)$$

Tự chấm cục bộ: `python score.py --pred submission.csv` (mặc định `--split public`).

## Baseline

`baseline.py`: với mỗi mẫu, tính **trung bình và độ lệch chuẩn theo thời gian** của 42 toạ độ (bỏ qua NaN)
→ 84 đặc trưng, chuẩn hoá rồi huấn luyện **LogisticRegression**. Đạt macro-F1 ≈ **0.583** trên tập ẩn
(0 điểm; ≈ 0.605 trên `test_public`). Baseline bỏ qua hoàn toàn thứ tự thời gian và dùng toạ độ thô chưa chuẩn hoá.

```
python baseline.py            # đọc data/, ghi submission.csv
python score.py --pred submission.csv
```
