# Bài B — Shipper đọc địa chỉ sau sáp nhập

- **Thời gian chạy:** 10 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không internet
- **Mô hình pretrained:** không được dùng (kể cả mô hình ngôn ngữ, LLM, bộ từ điển/danh mục hành chính bên ngoài)
- **Điểm baseline:** score = 0.930
- **Điểm BTC (100 điểm):** score = 0.995

## Bối cảnh

Một công ty giao hàng phục vụ một vùng gồm 16 tỉnh vừa được sắp xếp lại đơn vị hành chính: các tỉnh gộp thành
**8 tỉnh mới**, **bỏ cấp huyện**, hơn 500 xã/phường/thị trấn cũ được sáp nhập thành khoảng 235 xã/phường mới.
Hệ thống phân tuyến của công ty chỉ hiểu địa chỉ theo **đơn vị mới**, nhưng người gửi vẫn gõ địa chỉ theo thói quen:
người ghi tên cũ, người ghi tên mới, người ghi nửa cũ nửa mới; gõ không dấu, viết tắt, sai chính tả, quên dấu phẩy,
chen cả tên người nhận, số điện thoại và lời nhắn vào giữa.

> **Lưu ý:** toàn bộ tên tỉnh, huyện, xã/phường trong bài là **hư cấu** (trùng tên với địa danh thật, nếu có, chỉ là
> ngẫu nhiên); tên đường là các tên đường phổ biến ở Việt Nam. Danh mục và quy tắc sáp nhập trong bài là giả lập,
> không mô tả quy định hành chính thực tế nào.

Dữ liệu huấn luyện là các đơn hàng của tháng đầu sau sáp nhập, đã được nhân viên chuẩn hoá tay. Một trưởng bưu cục
dặn: *"Tên phường trên đơn có khi là tên cũ, có khi là tên mới — mà tên cũ chỗ này lại trùng tên mới chỗ khác.
Có xã cũ bị chia đôi, nửa này về xã A, nửa kia về xã B, tuỳ con đường. Con đường là thứ ít đổi nhất. Còn khách mới thì
gõ đủ kiểu, máy nào bộ gõ nấy."*

## Nhiệm vụ

Với mỗi địa chỉ `dia_chi`, hãy trả về 4 trường chuẩn hoá theo **đơn vị hành chính mới**:

| Trường | Ý nghĩa | Ví dụ |
|---|---|---|
| `so_nha` | số nhà, bỏ tiền tố (`số`, `số nhà`, `nhà số`, `SN`…), chữ cái viết hoa | `12`, `49C`, `45/7`, `200/29C` |
| `duong` | tên đường, đúng như cột `ten_duong` trong danh mục | `Nguyễn Văn Trỗi`, `Quốc lộ 1A`, `Đường số 5` |
| `xa_moi` | xã/phường **mới** chứa địa chỉ, đúng như cột `ten_xa_moi` | `Phường Hải Vân` |
| `tinh_moi` | tỉnh/thành phố **mới**, đúng như cột `ten_tinh_moi` | `Tỉnh Sa Mai`, `Thành phố Cát Hảo` |

Quy ước:
- Mỗi địa chỉ có đúng một số nhà và một tên đường; số nhà là số đứng ngay trước tên đường (có thể kèm tiền tố).
  Các con số khác — tổ, khu phố, ấp, thôn, lầu/tầng, số điện thoại, số thứ tự phường/quận cũ — **không** phải số nhà.
- Một địa chỉ có thể thiếu tỉnh, thiếu huyện, thậm chí thiếu cả xã/phường. Khi tạo dữ liệu, BTC đã kiểm tra rằng
  thông tin còn lại trong chuỗi **kết hợp với danh mục** đủ để xác định duy nhất đáp án (chỉ trừ số rất ít mẫu bị gõ
  sai quá nặng).

## Dữ liệu

| Thư mục | Số mẫu | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/gazetteer/` | danh mục | — | tra cứu (dùng chung cho mọi tập, kể cả tập chấm ẩn) |
| `data/train/` | 8 000 | ✅ 4 cột đáp án | huấn luyện |
| `data/test_public/` | 1 500 | ✅ `answers.csv` (bản sao dev) | tự chấm cục bộ |
| tập chấm ẩn | 4 000 | ❌ | chấm chính thức — chỉ có `test.csv`, cùng định dạng và phân phối với `test_public` |

```
data/gazetteer/tinh_moi.csv     # ma_tinh_moi, ten_tinh_moi                       (8 tỉnh mới)
data/gazetteer/xa_moi.csv       # ma_xa_moi, ten_xa_moi, ma_tinh_moi              (235 xã/phường mới)
data/gazetteer/duong.csv        # ma_xa_moi, ten_duong                            (1 595 dòng: đường thuộc xã mới)
data/gazetteer/xa_cu.csv        # ma_xa_cu, ten_xa_cu, ten_huyen_cu, ten_tinh_cu  (536 xã/phường/thị trấn cũ)
data/gazetteer/sap_nhap.csv     # ma_xa_cu, ma_xa_moi, pham_vi                    (bảng ánh xạ cũ -> mới)
data/train/train.csv            # id, dia_chi, so_nha, duong, xa_moi, tinh_moi
data/test_public/test.csv       # id, dia_chi
data/test_public/answers.csv    # id, so_nha, duong, xa_moi, tinh_moi  (bản sao dev — KHÔNG có trong tập chấm ẩn)
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.

Về danh mục:
- `sap_nhap.csv`: mỗi xã cũ ánh xạ tới một xã mới (`pham_vi = toàn bộ`) hoặc bị **chia** cho nhiều xã mới
  (`pham_vi = một phần`). Với xã bị chia, mỗi con đường của xã cũ thuộc đúng một trong các xã mới đó — tra theo `duong.csv`.
- Một tên đường có thể xuất hiện ở nhiều xã, nhiều tỉnh; trong cùng một xã mới, mỗi tên đường xuất hiện một lần.
- Tên xã/huyện có thể trùng nhau giữa các địa phương; tên trong danh mục dùng kiểu đặt dấu cũ (`Hoà`, `Thuỷ`).

Tập chấm ẩn do BTC giữ, chỉ gồm một file `test.csv` cùng định dạng `test_public/test.csv` (không có `answers.csv`, không có
`sample_submission.csv`); lúc chấm nó thay thế `data/test_public/`, nên code không được phụ thuộc vào hai file kia.
`test_public` phản ánh đúng kiểu dữ liệu của tập ẩn — và **không** giống hệt train: hãy so sánh kỹ hai tập.

## Định dạng nộp

```csv
id,so_nha,duong,xa_moi,tinh_moi
pub_000000,12A,Nguyễn Văn Trỗi,Phường Hải Vân,Tỉnh Sa Mai
pub_000001,45/7,Quốc lộ 1A,Xã Phú An,Tỉnh Tuy Gia
```

- File CSV UTF-8 có dòng tiêu đề; mỗi `id` của `test.csv` xuất hiện đúng một lần; mỗi trường tối đa 200 ký tự (dài hơn → bài nộp bị từ chối).
- Thiếu/trùng/thừa `id`, thiếu cột → bài nộp **bị từ chối**. Ô để trống được chấp nhận nhưng tính là sai.

## Ràng buộc

- Chỉ dùng dữ liệu được cấp (danh mục + `train/`). **Không** dùng dữ liệu ngoài, danh mục hành chính thật, mô hình
  pretrained, LLM hay API.
- **Không dùng đáp án `test_public/answers.csv` để huấn luyện, tra cứu hay viết luật riêng cho từng mẫu** — chỉ để tự chấm.
- Pipeline chạy xong trong 10 phút. Nộp kèm code và báo cáo kỹ thuật ngắn (các bước chuẩn hoá, cách khớp danh mục,
  cách xử lý địa chỉ cũ, kết quả validation).

## Chấm điểm

Mỗi trường được so khớp **chính xác** sau khi chuẩn hoá Unicode NFC, bỏ khoảng trắng ở hai đầu và gộp khoảng trắng
liên tiếp (phân biệt hoa/thường, phân biệt dấu). Điểm chính là độ chính xác trung bình trên 4 trường:

$$\text{score} = \frac{1}{4}\sum_{f \in \{\texttt{so\_nha},\, \texttt{duong},\, \texttt{xa\_moi},\, \texttt{tinh\_moi}\}} \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[\hat y_{i,f} = y_{i,f}\big], \qquad N = 4\,000$$

- Mỗi trường nặng như nhau; đúng số nhà và tên đường nhưng sai xã/tỉnh chỉ được 0.5 điểm thô cho mẫu đó.
- `score.py` báo thêm độ chính xác từng trường và tỉ lệ đúng **cả 4 trường** (`full_address_acc`) để tham khảo.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{score} - 0.930}{0.995 - 0.930},\ 0,\ 1\right)$$

Vì baseline đã khá mạnh nên thang quy đổi **dốc**: khoảng từ baseline đến điểm BTC chỉ rộng 0.065 điểm thô, tức 0.01 điểm thô ≈ 15 điểm. Sai số lấy mẫu của
điểm chấm trên 4 000 mẫu cỡ ±4 điểm; trên `test_public` (1 500 mẫu) cỡ ±6 điểm, và `test_public` hơi dễ hơn tập ẩn với các lời giải gần baseline.

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py` **không dùng `train/`**: nó chỉ đọc danh mục và `test.csv`. Với mỗi dòng của `duong.csv` (một cặp đường – xã mới) nó dựng một "tài liệu" gồm tên
đường (lặp hai lần), tên xã mới, tên tỉnh mới và tên mọi xã, huyện, tỉnh **cũ** được sáp nhập vào xã đó (theo `sap_nhap.csv` và `xa_cu.csv`); tất cả được bỏ dấu, đổi
sang chữ thường rồi biểu diễn bằng TF-IDF n-gram ký tự (độ dài 2–4). Mỗi địa chỉ (đã xoá số điện thoại bằng biểu thức chính quy, cũng bỏ dấu và chữ thường) được gán
cho tài liệu có cosine lớn nhất; từ đó suy ra `duong`, `xa_moi` và `tinh_moi` (tỉnh lấy theo xã), còn `so_nha` là token số đầu tiên có dạng số nhà. Điểm **0.930** trên
tập ẩn (số nhà đúng ≈ 93%, đường ≈ 97%, xã mới ≈ 87%, tỉnh mới ≈ 94%, đúng cả 4 trường ≈ 81%). Baseline chọn xã chỉ theo độ giống chuỗi tổng thể nên dễ nhầm khi
tên xã bị gõ sai nặng, khi địa chỉ ghi số thứ tự phường/quận hoặc thiếu hẳn tên xã, và lấy số nhà là số đầu tiên trong chuỗi.
Điểm BTC **0.995** đạt được bằng chương trình cổ điển (chuẩn hoá + khớp mờ + suy luận theo danh mục, CPU, vài giây); tối đa của metric là 1.0.
