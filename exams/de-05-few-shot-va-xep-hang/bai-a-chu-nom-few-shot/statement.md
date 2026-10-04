# Bài A — Đọc chữ Nôm của người chép mới

- **Thời gian chạy:** 10 phút (wall-clock) cho toàn bộ huấn luyện + suy luận lúc chấm
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (mọi trọng số phải học từ dữ liệu BTC cấp)
- **Điểm baseline:** accuracy trung bình theo episode = 0.642
- **Điểm BTC (100 điểm):** accuracy trung bình theo episode = 0.775

## Bối cảnh

Viện Nghiên cứu Hán Nôm vừa nhận về một rương sắc phong và gia phả từ các làng ven sông Đáy. Phần
lớn văn bản dùng những chữ "kiểu Nôm" chưa từng được số hoá: mỗi chữ ghép từ một **bộ thủ** quen
thuộc với một **thành phần biểu âm**, đôi khi kèm **dấu nháy** nhỏ ở góc trên trái. Kho mẫu chữ của
Viện có vài trăm chữ đã biết, nhưng các chữ trong rương lại là chữ **mới**. Với mỗi tập văn bản, chuyên
gia chỉ kịp khoanh tay **5 mẫu cho mỗi chữ**, phần còn lại bạn phải đọc giúp. Tệ hơn, mỗi tập do một
người chép khác viết — nét đậm nhạt, nghiêng, loang mực, giấy có đường kẻ và mực thấm từ mặt sau.

## Nhiệm vụ

Bài toán **phân loại few-shot 20-way 5-shot**. Dữ liệu kiểm tra được chia thành các *episode*. Mỗi
episode gồm:

- **support**: 20 lớp chữ × 5 ảnh có nhãn cục bộ `0..19`;
- **query**: 20 lớp × 5 ảnh cần dự đoán nhãn cục bộ.

Các lớp trong episode **không xuất hiện trong tập train** (lớp mới hoàn toàn), nhưng được ghép từ cùng
kho bộ thủ / thành phần / dấu nháy. Lớp của tập chấm ẩn cũng **khác** lớp của `test_public`. Nhãn cục
bộ chỉ có nghĩa trong episode của nó (nhãn `3` của episode 0 không liên quan gì nhãn `3` của episode 1).
Mỗi episode do **một người chép** viết toàn bộ.

## Dữ liệu

Ảnh xám 32×32 (`uint8`, nền giấy sáng, mực tối), lưu dạng `.npz`.

| Thư mục | Nội dung | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 400 lớp cơ sở × 20 ảnh = 8 000 ảnh | ✅ (`labels`, có ~2% nhãn nhiễu) | học biểu diễn / metric |
| `data/test_public/` | 40 episode (4 000 support + 4 000 query) trên 150 lớp mới | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm |
| tập chấm ẩn | 200 episode trên 300 lớp mới khác | — | **chấm chính thức** |

```
data/train/data.npz          # images (8000,32,32) uint8 · labels (8000,) int16 — id lớp cơ sở 0..399
data/test_public/data.npz    # support_images (N_s,32,32) · support_episode (N_s,) · support_label (N_s,) 0..19
                             # query_images (N_q,32,32)   · query_episode (N_q,)   · query_id (N_q,) "pub_00017"
data/test_public/answers.csv # query_id,episode,label   — bản sao dev, KHÔNG có trong tập chấm ẩn
```

**Lấy dữ liệu:** tại thư mục bài chạy `unzip public_data.zip` (tạo `data/train/` và `data/test_public/`).
Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` (id dạng `prv_00000`), không có đáp án. Lúc chấm,
`data/test_public/` được thay bằng tập ẩn.

Lưu ý phân phối: phong cách viết trong các episode test (nét dày hơn, nghiêng hơn, loang mực, đường kẻ
và mực thấm nhiều hơn) **lệch so với train**; `test_public` phản ánh đúng độ lệch này.

## Định dạng nộp

File CSV `submission.csv`, header `query_id,label`, mỗi query đúng một dòng, `label` là số nguyên `0..19`:

```csv
query_id,label
pub_00000,7
pub_00001,12
pub_00100,3
```

Thiếu / trùng / thừa `query_id`, giá trị rỗng hoặc nhãn ngoài `[0,19]` → bài nộp bị từ chối (lỗi định dạng).

## Ràng buộc

- Chỉ dùng dữ liệu trong `data/train/` và support của chính episode đang dự đoán (cùng query của episode,
  nếu muốn suy luận *transductive*). Không dùng dữ liệu ngoài, không dùng pretrained weights / LLM / API.
- **Không** dùng đáp án `test_public/answers.csv` để huấn luyện — nó chỉ để tự chấm.
- Huấn luyện + suy luận phải chạy xong trong 10 phút trên 4 CPU.

## Chấm điểm

Với episode $e$ có tập query $Q_e$ ($|Q_e| = 100$), độ chính xác là

$$\text{acc}_e = \frac{1}{|Q_e|}\sum_{q \in Q_e} \mathbb{1}\left[\hat y_q = y_q\right],
\qquad \text{Điểm thô} = \frac{1}{E}\sum_{e=1}^{E} \text{acc}_e ,\quad E = 200 \text{ (tập ẩn)}.$$

- Đoán ngẫu nhiên: ≈ 0.05. `score.py` báo thêm khoảng tin cậy 95% theo episode (`ci95`).
- Quy ra điểm 0–100: `Threshold(baseline=0.642, target=0.775)`,

$$\text{Điểm} = 100\cdot\text{clip}\left(\frac{\text{Điểm thô}-0.642}{0.775-0.642},\,0,\,1\right).$$

Tự chấm: `python score.py --pred submission.csv` (mặc định `--split public`).

## Baseline

`baseline.py`: với mỗi episode, chuẩn hoá từng ảnh (đảo về "mực" theo nền, căn trọng tâm mực về giữa,
trừ trung bình, chia chuẩn L2), lấy **tâm pixel** của 5 ảnh support mỗi lớp và gán query cho tâm gần
nhất. Không dùng tập train. Điểm thô: **0.642** trên tập ẩn (≈ 0.619 trên `test_public`).

Điểm BTC **0.775** đạt được hoàn toàn bằng CPU/scikit-learn trong vài phút, bằng cách học một không gian
nhúng trên 400 lớp train rồi phân loại theo tâm lớp. Đây không phải cận trên — tối đa là 1.0.
