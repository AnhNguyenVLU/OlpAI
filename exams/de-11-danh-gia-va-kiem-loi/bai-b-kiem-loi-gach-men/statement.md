# Bài B — Kiểm lỗi gạch men

- **Thời gian chạy:** 15 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (mọi tham số phải học từ dữ liệu BTC cấp)
- **Điểm baseline:** điểm thô = 0.413
- **Điểm BTC (100 điểm):** điểm thô = 0.795

## Bối cảnh

Một nhà máy gạch ốp lát ở miền Đông Nam Bộ sản xuất năm dòng gạch men. Cuối dây chuyền, camera chụp từng viên gạch và
máy tính phải báo viên nào có lỗi bề mặt để công nhân loại ra. Bộ phận chất lượng đã tích luỹ một kho ảnh gạch đạt chuẩn
và một "thư viện lỗi" nhỏ, tất cả chụp ở **xưởng cũ** dưới hệ đèn chiếu đều. Tháng này nhà máy chuyển dây chuyền sang
**xưởng mới**; mô hình của bạn sẽ chạy trên ảnh chụp ở xưởng mới.

Trưởng ca QC dặn: *"Đèn ở xưởng mới không đều như xưởng cũ: chỗ sáng chỗ tối, có lúc một dải đèn hắt ngang viên gạch,
ảnh cũng nhiễu hơn. Camera mới lắp, chưa ai lau ống kính. Thư viện lỗi chỉ có mấy kiểu lỗi hay gặp ở lò cũ — lò mới
thì đủ thứ lỗi lạ. Còn nữa: cả ca chỉ đặt **một ngưỡng chung** cho mọi dòng gạch, dòng nào báo động giả nhiều là
công nhân bỏ qua hết."*

## Nhiệm vụ

Với mỗi ảnh gạch trong tập test, hãy cho:

1. một **điểm bất thường** `score` (càng lớn càng nghi có lỗi);
2. **toạ độ** `(x, y)` của điểm ảnh nghi lỗi nhất (`x` = chỉ số cột, `y` = chỉ số hàng).

## Dữ liệu

Mọi ảnh là ảnh xám `uint8` kích thước **64×64**, chụp vuông góc một vùng bề mặt viên gạch. Năm dòng sản phẩm
(cột `mau_gach`):

| `mau_gach` | Mô tả bề mặt bình thường |
|---|---|
| `van_da` | vân đá: nền loang nhẹ + các đường vân mảnh tối (vân chính và vân phụ), hướng vân tuỳ viên |
| `van_go` | vân gỗ: sọc cong song song, chu kỳ và độ cong tuỳ viên, có các vệt lỗ thớ ngắn chạy dọc thớ |
| `granite` | đốm granite: rất nhiều đốm tối/sáng nhiều cỡ, nhiều độ đậm |
| `hoa_van` | hoa văn lặp: ô lặp hình học (vòng tròn, hình thoi, chữ thập, chấm + viền ô), chu kỳ, pha và góc xoay nhỏ tuỳ viên; có hạt men rải rác |
| `men_tron` | men trơn có hạt: nền gần phẳng với hạt men chấm nhỏ rải rác |

Biến thiên tự nhiên giữa các viên cùng dòng là đáng kể (pha, tỉ lệ, độ tương phản, hướng vân, độ sáng nền).

| Thư mục | Nội dung | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 1 560 ảnh xưởng cũ: 1 500 ảnh **bình thường** (300 ảnh/dòng) + **thư viện lỗi** 60 ảnh (4 loại × 15) | ✅ `loai_loi` + mask lỗi | huấn luyện |
| `data/test_public/` | 800 ảnh xưởng mới (160 ảnh/dòng), 305 ảnh có lỗi | ✅ `answers.csv` + `answers_masks.npz` (bản sao dev) | tự chấm cục bộ |
| tập chấm ẩn | 1 600 ảnh xưởng mới (320 ảnh/dòng) | ❌ | chấm chính thức — cùng định dạng và kiểu phân phối với `test_public` |

```
data/train/train.npz                  # images (1560,64,64) uint8; masks (1560,64,64) uint8 0/1; ids; mau_gach; loai_loi
data/train/train.csv                  # id, mau_gach, loai_loi   (cùng thứ tự với train.npz)
data/test_public/test.npz             # images (800,64,64) uint8; ids; mau_gach
data/test_public/test.csv             # id, mau_gach             (cùng thứ tự với test.npz)
data/test_public/answers.csv          # id, co_loi, loai_loi, nhom_loi — bản sao dev, KHÔNG có trong tập chấm ẩn
data/test_public/answers_masks.npz    # ids; masks (800,64,64) uint8 0/1 — bản sao dev
data/test_public/sample_submission.csv
```

Trong `answers.csv`: `co_loi` = 1 nếu ảnh có lỗi; `loai_loi` = tên loại lỗi (`khong_loi` với ảnh bình thường);
`nhom_loi` = `da_biet` (loại có trong thư viện) / `chua_thay` (loại không có trong thư viện) / rỗng (ảnh bình thường).
Đọc npz bằng `np.load(..., allow_pickle=False)`. Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.
Tập chấm ẩn do BTC giữ, cùng định dạng `test_public/test.npz` + `test.csv` (id dạng `prv_000000`) nhưng không có đáp án;
lúc chấm nó thay thế `data/test_public/`.

**Lỗi bề mặt.** Mỗi ảnh có lỗi có **đúng một** lỗi, có thể nằm ở bất kỳ đâu trong ảnh, **kể cả sát mép ảnh** (khung chụp
có thể gồm cả mép viên gạch); `masks` đánh dấu vùng lỗi (ảnh bình thường có mask toàn 0,
`loai_loi = khong_loi`). Thư viện lỗi trong train gồm 4 loại:

| `loai_loi` | Mô tả | Dòng gạch |
|---|---|---|
| `lan_tap` | lẫn tạp: đốm tối nhỏ có viền sáng mờ | mọi dòng (3 ảnh/dòng) |
| `loang_men` | loang men: một vùng biên mềm sáng hơn hoặc tối hơn nền một chút | mọi dòng (3 ảnh/dòng) |
| `xuoc` | xước: vạch thẳng mảnh, sáng | mọi dòng (3 ảnh/dòng) |
| `sai_hoa_van` | sai hoa văn: một ô của hoa văn bị lệch pha hoặc xoay | chỉ `hoa_van` (15 ảnh) |

Độ tương phản của lỗi nhìn chung **nhỏ** và phụ thuộc dòng gạch (ví dụ một đốm tối trên `granite` rất khó thấy).

**Lưu ý phân phối.** Ảnh test chụp ở xưởng mới: chiếu sáng không đều (gradient, dải sáng/tối, tối góc), phơi sáng
thay đổi, nhiễu cảm biến cao hơn, ảnh có thể hơi nhoè; `test_public` phản ánh đúng kiểu dịch chuyển này (tập chấm ẩn
gồm các phiên chụp khác, cùng kiểu). Khoảng **38%** ảnh test có lỗi, tỉ lệ này **như nhau ở mọi dòng gạch**. Khoảng
**40%** ảnh lỗi trong test thuộc các loại lỗi **không có trong thư viện**: `test_public` có 3 loại như vậy (tên loại
ghi trong `answers.csv`, cột `nhom_loi = chua_thay`); tập chấm ẩn có các loại chưa thấy **khác** (không trùng với 3 loại
của `test_public`). Phần còn lại là các loại trong thư viện (`nhom_loi = da_biet`).

## Định dạng nộp

File CSV `submission.csv`, header `id,score,x,y`, mỗi ảnh test đúng một dòng:

```csv
id,score,x,y
pub_000000,0.8315,41.0,17.0
pub_000001,-1.2034,12.5,60.0
```

- `score`: số thực bất kỳ (lớn = nghi lỗi), không NaN/inf.
- `x`, `y`: số thực với $0 \le x < 64$, $0 \le y < 64$ (`x` là cột, `y` là hàng của mảng ảnh `images[i, y, x]`).
- Thiếu / trùng / thừa id, giá trị không phải số, NaN/inf hoặc toạ độ ngoài ảnh → bài nộp **bị từ chối** (score.py báo lỗi).

## Ràng buộc

- Chỉ dùng dữ liệu BTC cấp. Không dữ liệu ngoài, không mô hình pretrained, không LLM/API.
- **Không dùng đáp án `test_public/answers.csv` hay `answers_masks.npz` để huấn luyện hay chọn tham số** — chỉ để tự chấm.
  Được dùng ảnh test **không nhãn** (của `test_public` và của tập chấm ẩn) theo bất kỳ cách nào: chuẩn hoá, học không giám
  sát, gán nhãn giả, thích nghi miền…
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 15 phút trên môi trường chấm.
- Nộp kèm code và báo cáo kỹ thuật ngắn.

## Chấm điểm

Gọi $D$ là tập ảnh có lỗi, $N$ là tập ảnh bình thường của tập chấm. Hai thành phần:

$$\mathrm{AUROC} = \Pr\big(\text{score}_i > \text{score}_j \;\big|\; i \in D,\ j \in N\big) \quad(\text{hoà tính } 1/2)$$

$$\mathrm{Pointing} = \frac{1}{|D|}\sum_{i \in D} \mathbf{1}\Big[\big(\lfloor x_i + 0.5\rfloor,\ \lfloor y_i + 0.5\rfloor\big) \in M_i^{(3)}\Big]$$

trong đó $M_i^{(3)}$ là mask lỗi của ảnh $i$ **giãn thêm 3 điểm ảnh** (mọi điểm ảnh cách vùng lỗi theo khoảng cách Euclid
$\le 3$); toạ độ sau khi làm tròn được kẹp vào $[0, 63]$. Điểm thô:

$$\text{điểm thô} = 0.7\cdot\mathrm{AUROC} + 0.3\cdot\mathrm{Pointing}$$

- AUROC tính **chung cho mọi dòng gạch** (không tách theo `mau_gach`): `score` của các dòng phải so sánh được với nhau,
  như một ngưỡng duy nhất của cả ca.
- Toạ độ của ảnh bình thường không ảnh hưởng điểm. `score` hằng số cho AUROC = 0.5; luôn chỉ vào tâm ảnh cho Pointing ≈ 0.1.
- `score.py` báo thêm `auroc`, `pointing` và các giá trị theo nhóm lỗi: `auroc_da_biet` / `auroc_chua_thay` (ảnh lỗi
  của nhóm đó so với **mọi** ảnh bình thường), `pointing_da_biet` / `pointing_chua_thay`.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{điểm thô} - 0.413}{0.795 - 0.413},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py`: bù chiếu sáng thô (chia ảnh cho bản làm mờ của chính nó), tính 18 đặc trưng **toàn ảnh** (trung bình,
độ lệch chuẩn, các phân vị, năng lượng gradient, năng lượng theo dải tần FFT), rồi `score` = khoảng cách Mahalanobis
(ước lượng hiệp phương sai co rút Ledoit–Wolf) tới các ảnh **bình thường cùng dòng** trong train; `(x, y)` luôn là tâm
ảnh `(32, 32)`. Nó không định vị được lỗi và gần như mù trước lỗi nhỏ: điểm thô **0.413** trên tập ẩn. Điểm BTC
**0.795** đạt được bằng mô hình cổ điển (numpy/scipy/scikit-learn, CPU, vài phút) trong giới hạn thời gian;
tối đa của điểm thô là 1.0.
