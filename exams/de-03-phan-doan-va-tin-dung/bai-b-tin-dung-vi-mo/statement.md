# Bài B — Tín dụng vi mô qua thời gian

- **Thời gian chạy:** 10 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** ROC-AUC = 0.68
- **Điểm BTC (100 điểm):** ROC-AUC = 0.79

## Bối cảnh

Một tổ chức tài chính vi mô phục vụ hộ nông dân và tiểu thương ở hơn 60 tỉnh thành đã giải ngân hàng chục nghìn
khoản vay nhỏ từ 2021. Ban điều hành muốn một mô hình chấm điểm để duyệt hồ sơ **từ giữa năm 2024 trở đi**.
Bộ phận dữ liệu gửi bạn bản trích xuất toàn bộ khoản vay 01/2021–06/2024 kèm kết quả trả nợ. Một chuyên viên
kỳ cựu dặn nhỏ: *"Dữ liệu này lấy thẳng từ hệ thống vận hành, cột nào cũng có lý do để tồn tại — nhưng
không phải cột nào cũng có sẵn vào lúc duyệt hồ sơ. Và thị trường 2024–2025 không giống 2021 đâu."*

## Nhiệm vụ

Với mỗi khoản vay giải ngân trong giai đoạn **07/2024–12/2025**, dự đoán xác suất khoản vay **vỡ nợ**
(`vo_no = 1`: quá hạn trên 90 ngày trong vòng đời khoản vay).

## Dữ liệu

| Thư mục | Số khoản vay | Giai đoạn giải ngân | Có nhãn? | Dùng để |
|---|---|---|---|---|
| `data/train/` | 46 000 | 01/2021 – 06/2024 | ✅ cột `vo_no` | huấn luyện |
| `data/test_public/` | 7 000 | 07/2024 – 12/2025 | ✅ `answers.csv` (bản sao dev) | tự chấm cục bộ |
| tập chấm ẩn | 20 000 | 07/2024 – 12/2025 | ❌ | chấm chính thức — cùng định dạng và phân phối với `test_public` |

```
data/train/train.csv                 # id, 30 đặc trưng, vo_no
data/test_public/test.csv            # id, 30 đặc trưng
data/test_public/answers.csv         # id, vo_no  (bản sao dev — KHÔNG có trong tập chấm ẩn)
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public/test.csv`, không có đáp án; lúc chấm nó thay thế `data/test_public/`.

Các cột (giá trị tiền: triệu VND):

| Nhóm | Cột |
|---|---|
| Khoản vay | `ngay_giai_ngan`, `san_pham`, `kenh`, `muc_dich`, `so_tien_vay`, `ky_han_thang`, `lai_suat_nam`, `gia_tri_tsdb` (giá trị tài sản đảm bảo), `co_bao_hiem`, `thoi_gian_xu_ly_ngay` |
| Địa bàn & vận hành | `ma_xa` (mã xã/phường, > 1 200 giá trị), `loai_khu_vuc`, `ma_can_bo` (cán bộ tín dụng phụ trách) |
| Khách hàng | `tuoi`, `gioi_tinh`, `hon_nhan`, `hoc_van`, `nghe_nghiep`, `so_nguoi_phu_thuoc`, `so_nam_cu_tru`, `xac_thuc_sdt` |
| Tài chính | `thu_nhap_thang`, `chi_tieu_thang`, `ty_le_no_thu_nhap` (khoản trả hằng tháng / thu nhập), `dien_tich_dat_m2`, `so_tien_tiet_kiem` |
| Lịch sử tín dụng | `so_khoan_vay_truoc`, `so_lan_tre_han_truoc`, `diem_cic` |
| Theo dõi | `so_lan_nhac_no` (số lần nhắc nợ, theo hệ thống tại ngày trích xuất) |

Ô trống là giá trị thiếu; **việc thiếu không xảy ra ngẫu nhiên**. Hãy kiểm tra kỹ từng đặc trưng: ý nghĩa,
thời điểm nó được ghi nhận, và phân phối của nó ở train so với test.

## Định dạng nộp

```csv
id,prob
pub_000000,0.0731
pub_000001,0.2410
```

- Mỗi `id` của `test.csv` xuất hiện đúng một lần; `prob` ∈ [0, 1], không NaN.
- Thiếu/trùng/thừa id hoặc giá trị không hợp lệ → bài nộp **bị từ chối**.

## Ràng buộc

- Chỉ dùng dữ liệu được cấp. **Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn mô hình** — chỉ để tự chấm.
- Pipeline chạy xong trong 10 phút. Nộp kèm code và báo cáo kỹ thuật ngắn (nêu rõ đặc trưng đã loại và lý do).

## Chấm điểm

Điểm chính là ROC-AUC trên tập chấm:

$$\mathrm{AUC} = \Pr\big(\hat p_i > \hat p_j \;\big|\; y_i = 1,\, y_j = 0\big) \quad(\text{hoà tính } 1/2)$$

- AUC chỉ phụ thuộc **thứ tự** xác suất: 0.5 = đoán ngẫu nhiên, 1.0 = xếp hạng hoàn hảo. Hiệu chỉnh (calibration) không ảnh hưởng điểm.
- `score.py` báo thêm **Brier score** $\frac1N\sum_i(\hat p_i - y_i)^2$ (càng nhỏ càng tốt) để tham khảo — dùng phân xử khi hoà điểm.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\mathrm{AUC} - 0.68}{0.79 - 0.68},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py`: LogisticRegression trên **mọi cột số** (impute median + StandardScaler), bỏ qua cột phân loại.
Điểm **0.68** trên tập ẩn. Điểm BTC **0.79** đạt được bằng mô hình cổ điển (scikit-learn, CPU, vài phút); tối đa của metric là 1.0.
