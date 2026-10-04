# Bài A — Ruộng lúa mùa khác

- **Thời gian chạy:** 15 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (mọi trọng số phải huấn luyện từ đầu trên `train/`)
- **Điểm baseline:** score = 0.31
- **Điểm BTC (100 điểm):** score = 0.57

## Bối cảnh

Sở Nông nghiệp một tỉnh đồng bằng sông Cửu Long muốn tự động thống kê diện tích lúa và số thửa ruộng
từ ảnh vệ tinh độ phân giải thấp. Nhóm kỹ thuật đã gán nhãn cẩn thận 1 500 ô ảnh chụp vào **vụ Đông Xuân**
(lúa đang thì con gái, xanh mướt, trời quang). Nhưng ảnh cần thống kê lại được chụp vào **vụ Hè Thu, lúc lúa
chín**: ruộng ngả vàng, nước kênh đục phù sa, cảm biến cân bằng màu khác và mây kéo đến nhiều hơn.
Không ai có thời gian gán nhãn lại. Mô hình của bạn phải "hiểu" ruộng là ruộng, dù mùa đã đổi.

## Nhiệm vụ

Mỗi mẫu là một ảnh RGB 64×64. Với mỗi ảnh, bạn phải:

1. **Phân đoạn ngữ nghĩa** từng pixel vào 4 lớp: `0` = ruộng lúa, `1` = kênh mương, `2` = nhà, `3` = cây xanh.
   Nhãn là mặt đất thật — kể cả pixel bị mây/bóng mây che phủ vẫn mang nhãn của vật thể bên dưới.
2. **Đếm số thửa ruộng**: số thành phần liên thông **4-láng giềng** (trên/dưới/trái/phải) của lớp `0`
   có **ít nhất 12 pixel** (hàm `count_fields` trong `baseline.py`). Các thửa được ngăn cách bởi kênh mương,
   nhà, cây; một kênh "cụt" không cắt hết thửa thì không tạo ra thửa mới.

## Dữ liệu

| Thư mục | Số ảnh | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 1 500 | ✅ `masks/` + `answers.csv` | huấn luyện (vụ Đông Xuân, ít mây) |
| `data/test_public/` | 400 | ✅ bản sao dev | tự chấm cục bộ (vụ Hè Thu, nhiều mây) |
| tập chấm ẩn | 400 | ❌ | chấm chính thức — cùng định dạng và phân phối với `test_public` |

```
data/train/images/tr_00000.png ...     # ảnh RGB 64x64
data/train/masks/tr_00000.png ...      # ảnh xám 64x64, giá trị pixel = nhãn 0..3
data/train/answers.csv                 # id,count  (số thửa ruộng)
data/test_public/images/pub_00000.png ...
data/test_public/masks/...             # bản sao dev — KHÔNG có trong tập chấm ẩn
data/test_public/answers.csv           # bản sao dev — KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip` (tạo ra `data/train/` và `data/test_public/`).

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` nhưng **chỉ có `images/`**. Lúc chấm, `data/test_public/`
được thay bằng tập ẩn, nên code phải liệt kê ảnh trong `images/` thay vì đọc `answers.csv`.
`test_public` phản ánh đúng kiểu dịch chuyển miền của tập ẩn.

## Định dạng nộp

File `submission.csv` với các cột `id,mask,count`, mỗi ảnh đúng một dòng:

```csv
id,mask,count
pub_00000,0:130 1:2 0:62 3:5 ...,7
pub_00001,0:4096,1
```

- `mask`: mã hoá run-length của **mask nhãn đã flatten theo hàng** (row-major: hàng 0 từ trái sang phải, rồi hàng 1, ...).
  Chuỗi gồm các token `nhan:do_dai` cách nhau bởi một dấu cách; tổng các `do_dai` phải bằng **4096**;
  `nhan` ∈ {0,1,2,3}. Hai run liền nhau được phép cùng nhãn. Hàm mã hoá mẫu: `rle_encode` trong `baseline.py`.
- `count`: số thửa ruộng dự đoán (chấp nhận số thực, nhưng nên là số nguyên).
- Thiếu id, trùng id, giá trị rỗng, RLE sai cú pháp/sai tổng độ dài → bài nộp **bị từ chối** (score.py báo lỗi).

## Ràng buộc

- Chỉ dùng dữ liệu trong `train/`. Không dữ liệu ngoài, không pretrained, không LLM/API.
- **Không dùng đáp án của `test_public` (`masks/`, `answers.csv`) để huấn luyện hay chọn ngưỡng** — chỉ để tự chấm.
  Được dùng ảnh test không nhãn để chuẩn hoá / thích nghi miền không giám sát lúc suy luận.
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 15 phút trên môi trường chấm.
- Nộp kèm code, trọng số (nếu có) và báo cáo kỹ thuật ngắn.

## Chấm điểm

Gọi $C$ là ma trận nhầm lẫn tích luỹ trên **toàn bộ pixel** của tập chấm ($N = 400$ ảnh). Với mỗi lớp $c$:

$$\mathrm{IoU}_c = \frac{C_{cc}}{\sum_j C_{cj} + \sum_i C_{ic} - C_{cc}}, \qquad \mathrm{mIoU} = \frac14 \sum_{c=0}^{3} \mathrm{IoU}_c$$

$$\mathrm{MAE} = \frac1N \sum_{i=1}^{N} \left|\hat n_i - n_i\right|, \qquad
\text{score} = 0.7\cdot \mathrm{mIoU} + 0.3\cdot\left(1 - \min\!\left(1, \frac{\mathrm{MAE}}{3}\right)\right)$$

Hành vi của metric:
- mIoU lấy trung bình **không trọng số** theo lớp: lớp hiếm (nhà ≈ 2% pixel, cây ≈ 7%) nặng ngang lớp ruộng (≈ 78%).
  Dự đoán "toàn ruộng" chỉ được mIoU ≈ 0.2.
- Phần đếm: sai trung bình 1 thửa → mất 0.1 điểm thô; sai trung bình từ 3 thửa trở lên → mất trọn 0.3.
  Nộp hằng số (trung vị train = 7) cho MAE ≈ 2.2. Mask nhiều đốm vụn hoặc kênh bị đứt (các thửa dính nhau) làm
  đếm sai nặng dù IoU khá.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{score} - 0.31}{0.57 - 0.31},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py` lấy mẫu 100 pixel/ảnh train, huấn luyện RandomForest trên giá trị **RGB thô** của từng pixel và
dự đoán từng pixel ảnh test; cột `count` là hằng số = trung vị số thửa trong train. Vì chỉ nhìn màu, nó sụp đổ khi
lúa và nước đổi màu: kênh mương gần như không nhận ra (IoU kênh ≈ 0.04). Điểm: **score = 0.31** trên tập ẩn
(mIoU ≈ 0.33, MAE ≈ 2.2). Điểm BTC **0.57** đạt được bằng mô hình cổ điển (scikit-learn, CPU) trong giới hạn thời gian;
đây không phải trần — tối đa là 1.0.
