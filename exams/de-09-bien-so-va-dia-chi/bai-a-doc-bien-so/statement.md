# Bài A — Camera trạm thu phí: đọc biển số

- **Thời gian chạy:** 20 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (kể cả mô hình OCR có sẵn như Tesseract; mọi trọng số phải huấn luyện từ đầu trên `train/`)
- **Điểm baseline:** score = 0.44
- **Điểm BTC (100 điểm):** score = 0.95

## Bối cảnh

Một trạm thu phí không dừng mới lắp camera hồng ngoại ở cửa ngõ quốc lộ. Đội vận hành có sẵn 3 000 ảnh biển số
đã gán nhãn, chụp từ hệ thống camera cũ ở các bãi đỗ xe trong thành phố, và cần một chương trình đọc biển số cho camera mới.
Một kỹ thuật viên lâu năm dặn: *"Trạm nằm trên trục xuyên Việt, xe ở đâu cũng chạy qua — đừng nghĩ mã tỉnh nào cũng đã
gặp. Ban đêm camera chuyển sang hồng ngoại. Còn biển số thì mỗi xưởng dập một kiểu chữ: nét đậm, nét mảnh, chữ bè,
chữ gầy, có số 4 hở đầu, số 7 có gạch ngang... Đừng học thuộc kiểu chữ của một xưởng."*

## Nhiệm vụ

Mỗi mẫu là một ảnh xám 128×64 chụp một biển số (biển nằm trọn trong ảnh nhưng không được cắt sát, có thể nghiêng,
méo phối cảnh, mờ, nhiễu, lóa, bóng đổ, bám bẩn hoặc bị che một phần). Hãy đọc **chuỗi ký tự** trên biển.

Biển số trong bài mô phỏng phong cách biển số Việt Nam ở dạng **đơn giản hoá** (không phản ánh đầy đủ quy định hiện hành;
mã tỉnh và sê-ri được sinh ngẫu nhiên):

| `loai_bien` | Bố cục | Ví dụ in trên biển | Chuỗi cần đọc |
|---|---|---|---|
| `o_to_1_dong` | ô tô, biển dài 1 dòng | `51F-123.45` | `51F12345` |
| `o_to_2_dong` | ô tô, biển vuông 2 dòng | `51F` / `1234` | `51F1234` |
| `xe_may` | xe máy, 2 dòng | `59-X1` / `234.56` | `59X123456` |

- 2 ký tự đầu là **mã tỉnh** (2 chữ số); tiếp theo là **sê-ri**: ô tô có 1 chữ cái (đôi khi 2 chữ cái), xe máy có
  1 chữ cái + 1 chữ số (1–9); cuối cùng là **số thứ tự** 4 hoặc 5 chữ số.
- Chữ cái dùng trên biển: `A B C D E F G H K L M N P S T U V X Y Z` (không có I, J, O, Q, R, W).
- Chuỗi cần đọc chỉ gồm chữ in hoa và chữ số, **bỏ** dấu `-`, dấu `.`, khoảng trắng và xuống dòng, đọc từ trái sang phải,
  từ dòng trên xuống dòng dưới.

## Dữ liệu

| Thư mục | Số ảnh | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 3 000 | ✅ `labels.csv` | huấn luyện (camera cũ) |
| `data/test_public/` | 600 | ✅ `answers.csv` (bản sao dev) | tự chấm cục bộ (camera trạm thu phí) |
| tập chấm ẩn | 2 000 | ❌ | chấm chính thức — cùng định dạng và phân phối với `test_public` |

```
data/train/images/tr_000000.png ...        # ảnh xám 8-bit 128x64
data/train/labels.csv                      # id, plate, hien_thi, loai_bien
data/test_public/images/pub_000000.png ...
data/test_public/answers.csv               # id, plate, hien_thi, loai_bien  (bản sao dev — KHÔNG có trong tập chấm ẩn)
data/test_public/sample_submission.csv
```

- `plate`: chuỗi cần đọc (đáp án). `hien_thi`: chuỗi in trên biển, các dòng ngăn bởi `|` (ví dụ `59-X1|234.56`).
  `loai_bien`: một trong ba bố cục ở bảng trên. Hai cột này chỉ để tham khảo khi huấn luyện; chúng **không** có ở tập ẩn.

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.

Tập chấm ẩn do BTC giữ, chỉ có `images/`. Lúc chấm, `data/test_public/` được thay bằng tập ẩn, nên code phải liệt kê ảnh
trong `images/` thay vì đọc `answers.csv`. `test_public` phản ánh đúng kiểu dịch chuyển miền của tập ẩn (cùng camera,
cùng các kiểu biến đổi), nhưng các xe — và các xưởng làm biển — là khác nhau.

## Định dạng nộp

```csv
id,plate
pub_000000,51F12345
pub_000001,59X123456
```

- Mỗi `id` của tập test xuất hiện đúng một lần. `score.py` tự bỏ khoảng trắng, `-`, `.`, `|` và chuyển chữ thường thành
  chữ hoa trước khi chấm; sau đó chuỗi phải khác rỗng, chỉ gồm `0-9`, `A-Z` và dài không quá 16 ký tự.
- Thiếu/trùng/thừa `id` hoặc giá trị không hợp lệ → bài nộp **bị từ chối**.

## Ràng buộc

- Chỉ dùng dữ liệu trong `train/`. Không dữ liệu ngoài, không mô hình/trọng số pretrained, không phông chữ hay ảnh biển số
  bên ngoài, không LLM/API.
- **Không dùng đáp án của `test_public` (`answers.csv`) để huấn luyện hay chọn ngưỡng** — chỉ để tự chấm.
  Được dùng ảnh test không nhãn để chuẩn hoá / thích nghi miền không giám sát lúc suy luận.
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 20 phút trên môi trường chấm.
- Nộp kèm code, trọng số (nếu có) và báo cáo kỹ thuật ngắn.

## Chấm điểm

Với mỗi ảnh $i$, gọi $y_i$ là chuỗi đúng, $\hat y_i$ là chuỗi dự đoán (đã chuẩn hoá như trên) và
$\mathrm{Lev}$ là khoảng cách Levenshtein (số phép chèn/xoá/thay ký tự tối thiểu):

$$\text{score} = \frac{1}{N}\sum_{i=1}^{N}\left(1 - \min\!\left(1,\ \frac{\mathrm{Lev}(\hat y_i, y_i)}{|y_i|}\right)\right), \qquad N = 2\,000$$

- Sai một ký tự trên biển 8 ký tự mất 0.125 điểm thô của ảnh đó; thiếu hay thừa ký tự cũng bị tính như vậy.
  Dự đoán quá dài hoặc lệch hoàn toàn được 0 cho ảnh đó (không âm).
- `score.py` báo thêm tỉ lệ đọc đúng **toàn bộ** biển (`exact_match`) để tham khảo.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{score} - 0.44}{0.95 - 0.44},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py`: nhị phân hoá cả ảnh bằng ngưỡng Otsu (giả định chữ tối trên nền sáng), lấy các thành phần liên thông có
kích thước giống ký tự, chia dòng theo toạ độ dọc, rồi phân loại từng thành phần (ảnh 10×16 điểm ảnh) bằng
LogisticRegression huấn luyện trên những ảnh train tách được đúng số ký tự. Nó không nắn ảnh nghiêng, không xử lý biển
chữ sáng nền tối, không tách được ký tự dính viền hay dính nhau và không dùng cấu trúc của biển số.
Điểm **0.44** trên tập ẩn (đọc đúng cả biển ≈ 0.13). Điểm BTC **0.95** đạt được bằng pipeline cổ điển
(scikit-learn, CPU) trong giới hạn thời gian; đây không phải trần — tối đa là 1.0.
