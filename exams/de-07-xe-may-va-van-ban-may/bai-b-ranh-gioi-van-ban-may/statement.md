# Bài B — Bóng ma trong kho lưu trữ

- **Thời gian chạy:** 10 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** score = 0.26
- **Điểm BTC (100 điểm):** score = 0.88

## Bối cảnh

Kho lưu trữ của một tỉnh vừa số hoá hàng nghìn bài viết địa phương: tin tức, truyện ngắn, nhật ký, bài giới thiệu du lịch…
Khi đối chiếu với bản gốc, thủ thư phát hiện nhiều bản số hoá "đọc lên vẫn trơn tru nhưng không phải của người viết":
từ một chỗ nào đó, phần còn lại đã bị một phần mềm cũ của kho — một mô hình thống kê đã học cách viết từ những văn bản
khác — viết nối thay cho nguyên văn. Không ai biết chỗ nối nằm ở đâu. Thủ thư dặn: *"Máy nói theo những gì nó đã đọc, nên
cứ viết một hồi là lạc đề, lẫn tên người, lặp cụm từ, và có mấy câu cửa miệng nghe rất 'sáo'. Nhưng người viết đôi lúc
cũng dùng y những câu ấy. Còn lô bài mới nhập kho có những thể loại và đề tài mà kho cũ chưa từng có — chớ vội tin hết những
gì học được từ kho cũ."*

## Nhiệm vụ

Mỗi mẫu là một văn bản tiếng Việt gồm **hai phần liên tiếp**: phần đầu do người viết, phần còn lại do máy viết nối tiếp
(máy được cho đọc phần đầu rồi viết tiếp). Mỗi văn bản có đúng một ranh giới. Với mỗi văn bản, dự đoán
`boundary_char_index` $b$ — số ký tự (tính theo ký tự Unicode, từ 0) của phần người viết:
`text[:b]` là phần người viết, `text[b:]` là phần máy viết (có thể bắt đầu bằng dấu cách, dấu câu hoặc xuống dòng).

- Mỗi phần có **ít nhất 120 từ** (tách theo khoảng trắng); cả văn bản dài khoảng 400–735 từ.
- Ranh giới có thể nằm đúng cuối một câu hoặc giữa một câu (tại ranh giới giữa hai từ).
- Hai phần được trình bày đồng nhất (cùng quy ước viết hoa đầu câu và xuống đoạn).

## Dữ liệu

| Thư mục | Số văn bản | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 1 500 | ✅ `answers.csv` | huấn luyện |
| `data/test_public/` | 800 | ✅ bản sao dev | tự chấm cục bộ — có thể loại và đề tài **không có trong `train`** |
| tập chấm ẩn | 800 | ❌ | chấm chính thức — cùng định dạng và kiểu dịch chuyển như `test_public`, nhưng là các văn bản khác |

```
data/train/data.csv                    # id,text
data/train/answers.csv                 # id,boundary_char_index
data/test_public/data.csv              # id,text
data/test_public/answers.csv           # bản sao dev — KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv # id,boundary_char_index
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip` (tạo ra `data/train/` và `data/test_public/`).

Cột `text` có chứa dấu phẩy và ký tự xuống dòng (dòng trống `\n\n` giữa các đoạn); file theo chuẩn CSV (bọc dấu nháy kép),
đọc bằng `pandas.read_csv(..., keep_default_na=False)` hoặc `csv.reader`. Văn bản đã chuẩn hoá Unicode NFC.

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public/data.csv`, **không có** `answers.csv`. Lúc chấm, `data/test_public/`
được thay bằng tập ẩn. `test_public` phản ánh đúng kiểu dịch chuyển miền của tập ẩn.

## Định dạng nộp

File `submission.csv` với các cột `id,boundary_char_index`, mỗi văn bản đúng một dòng:

```csv
id,boundary_char_index
pub_000000,1204
pub_000001,2310
```

- Mỗi `id` của `data.csv` xuất hiện đúng một lần; `boundary_char_index` là **số nguyên** (giá trị nguyên viết dạng `1204.0` vẫn hợp lệ)
  trong `[0, len(text)]`.
- Thiếu/trùng/thừa `id`, giá trị rỗng, không phải số nguyên hoặc ngoài khoảng → bài nộp **bị từ chối** (score.py báo lỗi).

## Ràng buộc

- Chỉ dùng dữ liệu trong `train/`. Không dữ liệu ngoài, không pretrained, không LLM/API.
- **Không dùng đáp án của `test_public` (`answers.csv`) để huấn luyện hay chọn siêu tham số** — chỉ để tự chấm.
  Được dùng văn bản test không nhãn (không có đáp án) để thích nghi miền lúc suy luận.
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 10 phút trên môi trường chấm.
- Nộp kèm code, trọng số (nếu có) và báo cáo kỹ thuật ngắn.

## Chấm điểm

Gọi $p_i$ là ranh giới dự đoán và $t_i$ là ranh giới thật của văn bản thứ $i$. Điểm mỗi văn bản giảm theo hàm mũ với khoảng
cách tính bằng ký tự, điểm thô là trung bình trên $N$ văn bản:

$$s_i=\exp\!\left(-\frac{|p_i-t_i|}{100}\right),\qquad \text{score}=\frac1N\sum_{i=1}^{N}s_i$$

- Trúng đúng ký tự: 1.00; lệch 25 ký tự: 0.78; lệch 50: 0.61; lệch 100: 0.37; lệch 300: 0.05; lệch 500: 0.007.
- Metric thưởng cho việc **đến gần**, không đòi trúng tuyệt đối; nhưng nộp "giữa văn bản" hoặc một tỉ lệ cố định chỉ được
  khoảng 0.15.
- `score.py` báo thêm `median_abs_err` (sai số trung vị, ký tự) và `within_50` (tỉ lệ văn bản lệch không quá 50 ký tự).

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{score} - 0.26}{0.88 - 0.26},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py` đặt ranh giới tại **cụm từ "sáo rỗng" đầu tiên** (trong danh sách 12 cụm cố định, ví dụ "Bên cạnh đó",
"đóng vai trò quan trọng", "góp phần") xuất hiện sau 120 từ đầu; nếu không có cụm nào thì đặt ở vị trí tỉ lệ ranh giới
trung bình của `train` × độ dài văn bản. Vì người viết cũng dùng các cụm này và máy không phải lúc nào cũng dùng, nó chỉ
đạt score ≈ 0.26 trên tập ẩn. Điểm BTC **0.88** đạt được bằng mô hình cổ điển (thống kê n-gram, numpy + scikit-learn,
CPU, vài phút); đây không phải trần — tối đa của metric là 1.0.
