# Bài A — Giờ cao điểm

- **Thời gian chạy:** 45 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (mọi trọng số phải huấn luyện từ đầu trên `train/`)
- **Điểm baseline:** AP@0.5 = 0.07
- **Điểm BTC (100 điểm):** AP@0.5 = 0.79

## Bối cảnh

Trung tâm điều hành giao thông của một thành phố lớn muốn đếm xe máy tại các ngã tư để điều chỉnh đèn tín hiệu
vào giờ cao điểm. Các camera giám sát cố định gửi về những khung hình nhỏ 128×96 điểm ảnh, nén JPEG. Nhóm kỹ thuật
đã gán nhãn 1 600 khung hình từ 12 camera, nhưng gần như tất cả chụp **ban ngày, trời quang**. Thứ cần xử lý lại là
khung hình **giờ tan tầm**: khoảng một nửa số ảnh chụp lúc trời tối hoặc đang mưa, từ những camera mà ta chưa gán nhãn
lần nào. Một kỹ sư vận hành dặn nhỏ: *"Ban đêm, đèn pha nổi bật hơn cả chiếc xe, vệt mưa cắt nát mọi đường nét, còn màu
áo màu mũ thì chẳng ai đoán trước được. Và đừng đếm xe đỗ trên vỉa hè hay xe đạp — chúng tôi chỉ quan tâm xe máy đang
có người ngồi."*

## Nhiệm vụ

Với mỗi ảnh, phát hiện **mọi xe máy có người lái** (một lớp duy nhất): trả về các hộp bao hình chữ nhật kèm điểm tin cậy.
Những thứ **không** phải mục tiêu: xe máy đỗ không có người ngồi, ô tô, xe đạp, người đi bộ (kể cả người gánh hàng rong),
cột điện, biển hiệu. Xe được chụp từ bên cạnh (trái/phải) hoặc từ phía trước/sau; có thể chở hai người; khi kẹt xe, các
xe che khuất lẫn nhau.

## Dữ liệu

| Thư mục | Số ảnh | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 1 600 | ✅ `answers.csv` | huấn luyện (12 camera, hầu hết ban ngày, trời quang) |
| `data/test_public/` | 400 | ✅ bản sao dev | tự chấm cục bộ (5 camera khác, khoảng một nửa ảnh ban đêm hoặc có mưa) |
| tập chấm ẩn | 800 | ❌ | chấm chính thức — cùng định dạng và kiểu dịch chuyển như `test_public`, nhưng là các camera và cảnh khác |

```
data/train/images/tr_000000.jpg ...        # ảnh RGB 128x96, JPEG
data/train/answers.csv                     # image_id,x1,y1,x2,y2,ignore
data/test_public/images/pub_000000.jpg ...
data/test_public/answers.csv               # bản sao dev — KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip` (tạo ra `data/train/` và `data/test_public/`).

`answers.csv` có mỗi dòng là một xe máy có người lái: `image_id`, toạ độ hộp bao theo điểm ảnh `x1, y1, x2, y2`
(gốc ở góc trên trái ảnh, `x` sang phải, `y` xuống dưới; số thực; hộp đã được cắt vào khung ảnh) và cờ `ignore`.
`ignore = 1` đánh dấu xe máy chỉ nhìn thấy dưới 50% diện tích (bị che khuất hoặc nằm ngoài khung hình): hộp này **không
được tính** khi chấm. Ảnh không có xe máy nào thì không xuất hiện trong `answers.csv`. Mỗi ảnh có 0–16 xe máy,
cao khoảng 12–43 điểm ảnh.

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` nhưng **chỉ có `images/`**. Lúc chấm, `data/test_public/` được thay
bằng tập ẩn, nên code phải liệt kê ảnh trong `images/` thay vì đọc `answers.csv`. `test_public` phản ánh đúng kiểu
dịch chuyển miền của tập ẩn.

## Định dạng nộp

File `submission.csv` với các cột `image_id,x1,y1,x2,y2,score`, **nhiều dòng cho mỗi ảnh** (mỗi dòng một hộp dự đoán):

```csv
image_id,x1,y1,x2,y2,score
pub_000000,10.0,40.0,36.0,64.0,0.91
pub_000000,50.5,52.0,62.0,78.0,0.35
pub_000001,3.0,48.0,29.5,80.0,0.77
```

- `image_id` phải thuộc tập chấm; ảnh không có dòng nào được hiểu là không phát hiện gì.
- `x2 > x1`, `y2 > y1`, mọi giá trị là số hữu hạn (không rỗng/NaN); tối đa **100 hộp mỗi ảnh**.
  Toạ độ nằm ngoài khung ảnh được cắt vào `[0, 128] × [0, 96]` trước khi chấm.
- `score` là số thực bất kỳ, càng lớn càng tin cậy (chỉ thứ tự giữa các hộp có ý nghĩa).
- Sai định dạng (thiếu cột, `image_id` lạ, hộp suy biến, quá 100 hộp/ảnh…) → bài nộp **bị từ chối** (score.py báo lỗi).

## Ràng buộc

- Chỉ dùng dữ liệu trong `train/`. Không dữ liệu ngoài, không pretrained, không LLM/API.
- **Không dùng đáp án của `test_public` (`answers.csv`) để huấn luyện hay chọn ngưỡng** — chỉ để tự chấm.
  Được dùng ảnh test không nhãn để chuẩn hoá / thích nghi miền lúc suy luận.
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 45 phút trên môi trường chấm.
- Nộp kèm code, trọng số (nếu có) và báo cáo kỹ thuật ngắn.

## Chấm điểm

Điểm chính là **AP@0.5** (Average Precision, một lớp) trên toàn bộ tập chấm, tính theo kiểu PASCAL VOC (nội suy mọi điểm):

1. Gộp tất cả hộp dự đoán của mọi ảnh, sắp theo `score` giảm dần (hoà thì giữ thứ tự trong file).
2. Với mỗi hộp dự đoán, tìm hộp đáp án **cùng ảnh** có IoU lớn nhất, $\mathrm{IoU}=\frac{|A\cap B|}{|A\cup B|}$:
   - nếu $\mathrm{IoU}\ge 0.5$ và đáp án đó là hộp bình thường chưa được ghép → **TP**, đánh dấu đáp án đã được ghép;
   - nếu $\mathrm{IoU}\ge 0.5$ nhưng đáp án đó đã được ghép trước đó → **FP** (phát hiện trùng lặp);
   - nếu $\mathrm{IoU}\ge 0.5$ và đáp án đó có `ignore = 1` → hộp dự đoán bị bỏ qua (không TP, không FP);
   - còn lại → **FP**.
3. Sau mỗi hộp (đã loại các hộp bỏ qua) tính precision $p_k=\mathrm{TP}_k/k$ và recall $r_k=\mathrm{TP}_k/N$,
   với $N$ là số hộp đáp án có `ignore = 0` của cả tập. Lấy đường bao precision rồi tính diện tích:

$$\mathrm{AP}=\sum_{k}(r_k-r_{k-1})\,\max_{j\ge k}p_j,\qquad r_0=0$$

`score.py` báo thêm **AP@0.75** (cùng cách tính, ngưỡng IoU 0.75) để tham khảo.

Hành vi của metric:
- Chỉ **thứ hạng** của `score` giữa các hộp có ý nghĩa. Thêm các hộp có điểm thấp hơn mọi hộp đã có không bao giờ làm AP giảm
  (chỉ có thể tăng recall); hộp sai mang điểm cao mới làm hỏng precision. Hộp lệch vị trí (IoU < 0.5) vừa là FP vừa không nhận được TP: định vị quan trọng
  ngang với nhận dạng.
- Một xe được phát hiện hai lần chỉ tính một TP — nhưng NMS quá mạnh sẽ xoá nhầm các xe sát nhau khi kẹt xe.
- Nộp hộp ngẫu nhiên hoặc hộp "đoán theo vị trí thường gặp" cho AP@0.5 dưới 0.07.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\mathrm{AP@0.5} - 0.07}{0.79 - 0.07},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py` trượt cửa sổ **vuông** hai cỡ (18 và 30 điểm ảnh) trên ảnh xám, thu mỗi cửa sổ về 10×10 điểm ảnh làm đặc trưng
cho LogisticRegression (dương = cửa sổ quanh một xe máy trong `train/`), rồi lọc bằng NMS (ngưỡng IoU 0.3). Vì chỉ nhìn
độ sáng thô trên cửa sổ vuông, nó gần như không phân biệt được xe máy với nền, ô tô hay người đi bộ: AP@0.5 ≈ 0.07
trên tập ẩn. Điểm BTC **0.79** đạt được bằng mô hình cổ điển (numpy + scikit-learn, CPU) trong giới hạn thời
gian; đây không phải trần — tối đa của metric là 1.0.
