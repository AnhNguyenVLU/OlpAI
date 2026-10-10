# Bài B — Gửi voucher cho ai?

- **Thời gian chạy:** 10 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** uplift@30% = 2.5 điểm phần trăm
- **Điểm BTC (100 điểm):** uplift@30% = 14.6 điểm phần trăm

## Bối cảnh

Một sàn thương mại điện tử Việt Nam chuẩn bị chiến dịch "Sale giữa tháng": gửi **voucher giảm 20%** cho đúng **30%**
khách hàng (ngân sách chỉ đủ chừng đó). Voucher tốn tiền, nên ban giám đốc muốn gửi cho những người mà voucher **thực sự
làm họ mua thêm** — không phải những người đằng nào cũng mua.

Phòng dữ liệu có hai nguồn. Thứ nhất là nhật ký chiến dịch năm ngoái: đội marketing **tự tay chọn** người nhận và mức giảm
(10%, 20% hoặc 30%) theo kinh nghiệm. Thứ hai là một thí nghiệm A/B nhỏ vừa chạy tháng trước trên 40 000 khách khác: tung
đồng xu để quyết định ai nhận voucher 20%. Chị trưởng nhóm phân tích dặn: *"Số liệu năm ngoái nhìn thì đẹp lắm — ai nhận
voucher cũng mua nhiều hơn hẳn — nhưng đó là vì marketing toàn chọn khách ruột. Đừng nhầm 'người hay mua' với 'người mua
vì voucher'. Năm ngoái mỗi người nhận một mức giảm khác nhau, năm nay chỉ có 20%. Và cẩn thận: có những khách nhận voucher
xong lại… mua ít đi."*

## Nhiệm vụ

Với mỗi khách hàng $i$ trong tập test, gọi **hiệu ứng nhân quả** của voucher 20% là

$$\tau_i = \Pr(\text{mua trong 14 ngày} \mid \text{nhận voucher 20\%}) - \Pr(\text{mua trong 14 ngày} \mid \text{không nhận voucher}).$$

Hãy cho mỗi khách một điểm `uplift` (càng lớn càng nên gửi). BTC sẽ gửi voucher cho 30% khách có điểm cao nhất; mục tiêu
là chọn nhóm có $\tau$ trung bình cao nhất. Chỉ **thứ tự** của điểm là quan trọng (không cần hiệu chỉnh về đúng thang xác suất).

## Dữ liệu

| Thư mục | Số khách | Có kết quả? | Dùng để |
|---|---|---|---|
| `data/train/` | 80 000 | ✅ `nhan_voucher`, `muc_giam`, `mua_hang` (chiến dịch năm ngoái — dữ liệu **quan sát**, người nhận do marketing chọn) | huấn luyện |
| `data/test_public/` | 40 000 | ✅ `answers.csv`: `nhan_voucher`, `mua_hang` (thí nghiệm **ngẫu nhiên**, voucher 20%, xác suất nhận 1/2) — bản sao dev | tự chấm cục bộ (ước lượng có nhiễu) |
| tập chấm ẩn | 40 000 | ❌ — BTC giữ hiệu ứng thật $\tau_i$ của từng khách | chấm chính thức — cùng định dạng và phân phối khách hàng với `test_public` |

```
data/train/train.csv                    # id, 21 đặc trưng, nhan_voucher, muc_giam, mua_hang
data/test_public/test.csv               # id, 21 đặc trưng
data/test_public/answers.csv            # id, nhan_voucher, mua_hang   (bản sao dev — KHÔNG có trong tập chấm ẩn)
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public/test.csv`; lúc chấm, BTC chạy lại mã nộp trong một thư mục chỉ có
`data/train/train.csv` và `data/test_public/test.csv` (ở đây `test.csv` là của tập ẩn; không có `answers.csv` và
`sample_submission.csv`). Vì dữ liệu là **tổng hợp**, BTC biết chính xác $\tau_i$ của từng khách trong tập ẩn và chấm
trực tiếp bằng nó. Ở `test_public` chỉ có kết quả quan sát được (mỗi khách hoặc nhận, hoặc không nhận voucher), nên điểm
tự chấm là một **ước lượng** (xem Chấm điểm).

Đặc trưng (đo **trước** chiến dịch; tiền tính bằng nghìn đồng):

| Nhóm | Cột |
|---|---|
| Nhân khẩu | `tuoi`, `gioi_tinh`, `vung` (`mien_bac`/`mien_trung`/`mien_nam`), `do_thi` (0/1) |
| Tài khoản | `thiet_bi` (`android`/`ios`/`web`), `kenh_thu_hut` (`tu_nhien`/`quang_cao`/`gioi_thieu`/`mang_xa_hoi`), `so_thang_tham_gia`, `hang_thanh_vien` (`dong` < `bac` < `vang` < `kim_cuong`), `lien_ket_vi` (đã liên kết ví điện tử) |
| Mua sắm | `so_don_90_ngay`, `gia_tri_don_tb`, `so_ngay_tu_lan_mua_cuoi`, `ty_le_dung_voucher` (tỉ lệ đơn trước đây có dùng voucher), `ty_le_hoan_tra`, `so_sp_trong_gio` (số sản phẩm đang nằm trong giỏ) |
| Tương tác | `so_phien_30_ngay`, `so_lan_xem_khuyen_mai_30_ngay`, `ty_le_mo_thong_bao`, `nhan_thong_bao` (đồng ý nhận thông báo), `so_khieu_nai_12_thang`, `diem_danh_gia_tb` (ô trống: chưa từng đánh giá) |
| Chỉ có ở train | `nhan_voucher` (0/1), `muc_giam` (0 nếu không nhận; 10, 20 hoặc 30 nếu nhận), `mua_hang` (mua trong 14 ngày sau chiến dịch, 0/1) |

Ở train, khoảng 59% khách được gửi voucher (theo lựa chọn của marketing); ở `test_public`, mỗi khách nhận voucher 20% với xác suất 1/2 (tung đồng xu, nên khoảng một nửa số khách).

## Định dạng nộp

```csv
id,uplift
pub_000000,0.1834
pub_000001,-0.0212
```

- Mỗi `id` của `test.csv` xuất hiện đúng một lần; `uplift` là số thực hữu hạn bất kỳ (được phép âm), không NaN.
- Thiếu/trùng/thừa id hoặc giá trị không hợp lệ → bài nộp **bị từ chối**; cột thừa (nếu có) bị bỏ qua.

## Ràng buộc

- Chỉ dùng dữ liệu được cấp. **Không dùng `test_public/answers.csv` để huấn luyện** (kể cả làm dữ liệu bổ sung, hiệu chỉnh
  hay stacking, **kể cả sao chép hoặc nhúng dữ liệu public vào mã nộp**) — chỉ để tự chấm. Được dùng đặc trưng (không nhãn)
  của `test.csv`.
- **Khi chấm, BTC chạy lại mã nộp trong thư mục chỉ có `train.csv` và `test.csv` của tập ẩn.** Vì vậy mã phải tự huấn luyện
  từ `train.csv`; phần nộp của bài này chỉ gồm mã nguồn (≤ 200 KB, không kèm CSV/NPY/pickle hay dữ liệu nhúng) và báo cáo.
  Mã phải chạy được bằng một lệnh duy nhất (ví dụ `python solution.py`; tên file do giáo viên quy định) và ghi `submission.csv`
  vào thư mục hiện tại; file dự đoán do BTC tạo khi chạy lại mã nên không cần nộp kèm. BTC rà soát mã của các đội đứng đầu.
- Pipeline (huấn luyện + suy luận) chạy xong trong 10 phút. Nộp kèm code và báo cáo kỹ thuật ngắn (nêu rõ cách xử lý việc
  người nhận voucher năm ngoái không được chọn ngẫu nhiên và việc có nhiều mức giảm).

## Chấm điểm

Gọi $N$ là số khách của tập chấm, $k = \mathrm{round}(0.3N)$, và $S$ là $k$ khách có `uplift` cao nhất (nếu nhiều khách
hoà điểm đúng tại ngưỡng, mỗi người được tính với trọng số bằng nhau sao cho tổng trọng số của $S$ đúng bằng $k$ — tương đương
bốc ngẫu nhiên trong nhóm hoà). Điểm thô, tính bằng **điểm phần trăm**:

$$\text{uplift@30\%} = 100\cdot\left(\frac1k\sum_{i\in S}\tau_i - \frac1N\sum_{i=1}^{N}\tau_i\right)$$

- **Tập chấm ẩn:** tính trực tiếp bằng $\tau_i$ thật.
- **`test_public`:** $\tau_i$ không quan sát được; `score.py` dùng ước lượng không chệch từ thí nghiệm ngẫu nhiên:

$$\widehat{\text{uplift@30\%}} = 100\cdot\Big[\big(\bar Y^{S}_{T=1} - \bar Y^{S}_{T=0}\big) - \big(\bar Y_{T=1} - \bar Y_{T=0}\big)\Big]$$

  trong đó $\bar Y^{S}_{T=1}$ là tỉ lệ mua của những khách trong $S$ đã nhận voucher, v.v. Ước lượng này **có nhiễu**:
  độ lệch chuẩn bootstrap ≈ 0.75 điểm phần trăm (≈ 6 điểm trên thang 100), được in ở trường `uplift_at_30_se`. Chênh lệch giữa hai lời giải
  trên `test_public` nhỏ hơn cỡ này chưa đủ để kết luận lời giải nào tốt hơn.

Hành vi của metric:
- Chọn ngẫu nhiên hoặc nộp hằng số cho uplift@30% = 0. Metric **có thể âm**: chọn trúng những người voucher làm họ mua ít
  đi, hoặc những người đằng nào cũng mua (hiệu ứng ≈ 0) trong khi bỏ sót người thực sự bị thuyết phục.
- Chọn hoàn hảo theo $\tau_i$ thật cho khoảng 22 điểm phần trăm trên tập ẩn (không đạt được trọn vẹn vì $\tau_i$ còn phụ thuộc
  những yếu tố không có trong dữ liệu).
- `score.py` báo thêm uplift@10% và uplift@50% để tham khảo.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{uplift@30\%} - 2.5}{14.6 - 2.5},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py` huấn luyện HistGradientBoosting dự đoán xác suất **mua hàng** $P(\text{mua\_hang}=1 \mid x)$ trên train — bỏ qua
việc khách có nhận voucher hay không — rồi gửi voucher cho những người "hay mua" nhất. Đây là sai lầm kinh điển: mô hình
chọn trúng nhiều khách đằng nào cũng mua. Điểm: **uplift@30% = 2.5** trên tập ẩn (chỉ nhỉnh hơn chọn ngẫu nhiên).
Điểm BTC **14.6** đạt được bằng mô hình cổ điển (scikit-learn, CPU, vài phút).
