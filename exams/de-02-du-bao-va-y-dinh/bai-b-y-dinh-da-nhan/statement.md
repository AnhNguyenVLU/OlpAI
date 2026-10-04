# Bài B — Tổng đài viên không ngủ

- **Thời gian chạy:** 10 phút (toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** CPU 4 nhân, 16 GB RAM, không có internet
- **Mô hình pretrained:** không được dùng (không word embedding, không PhoBERT/LLM, không API ngoài)
- **Điểm baseline:** ½(micro-F1 + macro-F1) = 0.755 (trên tập chấm ẩn)
- **Điểm BTC (100 điểm):** ½(micro-F1 + macro-F1) = 0.875

## Bối cảnh

Một sàn thương mại điện tử Việt Nam nhận hàng chục nghìn tin nhắn chăm sóc khách hàng mỗi ngày.
Đội vận hành muốn một bộ định tuyến tự động: đọc tin nhắn và gắn **tất cả** các ý định có trong đó
(một khách vừa đòi huỷ đơn vừa hỏi hoàn tiền phải được chuyển cho cả hai bộ phận). Dữ liệu lịch sử
được trích từ hệ thống ticket cũ — nơi nhân viên gõ tin mẫu có dấu đầy đủ và đôi khi gắn nhãn vội.
Còn khách hàng thật thì gõ trên điện thoại: *"shop oi huy dum e don DH123456 r hoan tien vs"*,
*"hangf giao sai mauf"*, *"k dc"*...

## Nhiệm vụ

Cho mỗi tin nhắn `text`, dự đoán **tập nhãn ý định** (1–3 nhãn trên tổng số 15 nhãn, danh sách ở
`train/labels.csv`):

| Nhãn | Ý nghĩa |
|---|---|
| `kiem_tra_don` | Hỏi tình trạng / theo dõi đơn hàng |
| `huy_don` | Yêu cầu huỷ đơn |
| `doi_tra` | Đổi / trả hàng |
| `hoan_tien` | Yêu cầu / hỏi về hoàn tiền |
| `loi_thanh_toan` | Lỗi thanh toán, bị trừ tiền |
| `giao_cham` | Giao hàng chậm / chưa nhận được |
| `doi_dia_chi` | Đổi địa chỉ / SĐT nhận hàng |
| `khieu_nai_nhan_vien` | Khiếu nại thái độ nhân viên / shipper |
| `khuyen_mai` | Hỏi mã giảm giá / khuyến mãi |
| `tai_khoan` | Tài khoản / đăng nhập / OTP |
| `san_pham_loi` | Sản phẩm lỗi / hỏng / sai mẫu |
| `hoi_san_pham` | Hỏi thông tin sản phẩm, còn hàng, size |
| `xuat_hoa_don` | Xuất hoá đơn VAT *(hiếm)* |
| `bao_hanh` | Bảo hành / sửa chữa *(ít)* |
| `khen_ngoi` | Khen ngợi / góp ý tích cực *(hiếm)* |

Lưu ý: lời kết lịch sự kiểu "cảm ơn shop" **không** phải là `khen_ngoi`.

## Dữ liệu

Giải nén tại thư mục bài: `unzip public_data.zip` (tạo thư mục `data/`).

| Thư mục | Số mẫu | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 12 000 | ✅ `answers.csv` (nhãn **có nhiễu ~8%**) | huấn luyện |
| `data/test_public/` | 3 000 | ✅ `answers.csv` (bản sao dev, nhãn sạch) | chạy pipeline, tự chấm cục bộ |

Tập chấm ẩn do BTC giữ (3 000 mẫu), cùng định dạng `test_public` nhưng không có `answers.csv`; lúc
chấm, `data/test_public/` được thay bằng tập ẩn. Tập ẩn **cùng kiểu phân phối với `test_public`** —
và cả hai đều **khác train**:

- train phần lớn viết có dấu chuẩn; test có tỉ lệ lớn tin **không dấu**, **teencode**
  (`k`, `ko`, `dc`, `j`, `r`, `mk`...), **lỗi gõ Telex** khi bộ gõ tắt (`dduowcj`, `hangf`), lỗi phím,
  viết hoa toàn bộ, bỏ dấu câu;
- một phần cách diễn đạt và tên sản phẩm ở test **không xuất hiện** trong train. Phần diễn đạt mới của
  tập ẩn **khác** phần diễn đạt mới của `test_public` — học thuộc `test_public` không giúp gì;
- nhãn train có ~8% mẫu bị thiếu / thừa / sai một nhãn; nhãn test sạch;
- không có tin nào ở test trùng nguyên văn với train.

```
data/train/data.csv          # id,text
data/train/answers.csv       # id,labels      (labels: các nhãn nối bằng '|', ví dụ "doi_tra|hoan_tien")
data/train/labels.csv        # label,description
data/test_public/data.csv    # id,text        (id dạng pub_00000; ở tập ẩn: prv_00000)
data/test_public/answers.csv # id,labels      — chỉ có ở bản sao dev
```

## Định dạng nộp

Code phải đọc `data/test_public/data.csv` và ghi `submission.csv`:

```csv
id,labels
pub_00000,hoan_tien|huy_don
pub_00001,kiem_tra_don
pub_00002,
```

- Mỗi `id` của tập test xuất hiện **đúng một lần**; nhãn phải thuộc `train/labels.csv`, nối bằng `|`.
- Cột `labels` được phép rỗng (dự đoán không có nhãn — thường bị phạt qua recall).
- Thiếu id, trùng id, id lạ hoặc nhãn lạ → bài nộp bị từ chối (score.py báo lỗi, exit code 1).

## Ràng buộc

- CPU, không internet, 10 phút cho cả huấn luyện + suy luận.
- **Không** dùng dữ liệu ngoài, trọng số pretrained, từ điển/bộ chuẩn hoá tải từ internet.
  Được tự viết luật chuẩn hoá (bỏ dấu, quy đổi teencode, giải Telex) dựa trên quan sát dữ liệu.
- **Không** dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn ngưỡng (chỉ để tự chấm).

## Chấm điểm

Với ma trận nhãn thật $Y$ và dự đoán $\hat Y$ ($N$ mẫu × $K=15$ nhãn), với mỗi nhãn $k$ tính
$TP_k, FP_k, FN_k$:

$$\text{micro-F1} = \frac{2\sum_k TP_k}{2\sum_k TP_k + \sum_k FP_k + \sum_k FN_k},\qquad
\text{macro-F1} = \frac{1}{K}\sum_{k=1}^{K}\frac{2TP_k}{2TP_k + FP_k + FN_k}$$

$$\text{score} = \tfrac12\left(\text{micro-F1} + \text{macro-F1}\right)$$

- micro-F1 bị chi phối bởi nhãn phổ biến; macro-F1 cho **mỗi nhãn hiếm cùng trọng số** như nhãn
  phổ biến — bỏ qua `xuat_hoa_don` hay `khen_ngoi` sẽ mất ~1/15 macro-F1.
- Nhãn không có trong cả thật lẫn dự đoán được tính F1 = 1 (không xảy ra với tập test chuẩn).

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\operatorname{clip}\!\left(\frac{\text{score} - 0.755}{0.875 - 0.755},\,0,\,1\right)$$

## Baseline

`baseline.py`: TF-IDF theo **từ** (1–2 gram) trên văn bản gốc + One-vs-Rest Logistic Regression,
ngưỡng 0.5 cho mọi nhãn. Điểm thô ≈ **0.755** trên tập ẩn (≈ 0.746 trên `test_public`) — 0 điểm.

```
python baseline.py                     # đọc data/, ghi submission.csv
python score.py --pred submission.csv  # chấm trên test_public
```
