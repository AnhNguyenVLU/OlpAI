# Bài A — Khen chê từng khía cạnh

- **Thời gian chạy:** 10 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (không word embedding, không PhoBERT/LLM, không API ngoài)
- **Điểm baseline:** micro-F1 = 0.725
- **Điểm BTC (100 điểm):** micro-F1 = 0.97

## Bối cảnh

Một công ty khởi nghiệp tổng hợp đánh giá khách sạn, homestay, resort trên khắp Việt Nam muốn hiển thị cho mỗi cơ sở
lưu trú một bảng "khen – chê" theo từng khía cạnh: vị trí, phòng ở, vệ sinh, phục vụ, giá cả, ăn uống, tiện ích, độ yên tĩnh.
Dữ liệu cũ đến từ ba ứng dụng đặt phòng lâu năm (`vivu`, `dimuon`, `nghiduong`) và đã được đội kiểm duyệt gắn nhãn.
Tháng này công ty ký thêm hợp đồng với `phuot` — ứng dụng đặt phòng của dân du lịch bụi, nơi khách gõ đánh giá bằng
điện thoại ngay trên đường đi.

Trưởng nhóm kiểm duyệt dặn: *"Đừng nhìn số sao mà đoán. Khách cho 3 sao vẫn có thể khen nhân viên hết lời rồi chê phòng
tắm; 'không có gì để chê' là khen, còn 'không sạch lắm' là chê. Câu nhắc tới hồ bơi chưa chắc đã là khen hay chê hồ bơi.
Và khách bên app mới gõ không dấu, viết tắt nhiều lắm đấy."*

Toàn bộ đánh giá là **dữ liệu tổng hợp** sinh từ ngữ pháp mẫu; tên cơ sở lưu trú là hư cấu.

## Nhiệm vụ

Với mỗi đánh giá, xác định **các khía cạnh được đánh giá** và **cực tính** của từng khía cạnh: `pos` (khen) hoặc `neg` (chê).
Có 8 khía cạnh:

| Mã | Khía cạnh | Gồm | Ví dụ |
|---|---|---|---|
| `vi_tri` | Vị trí, cảnh quan | vị trí, địa điểm, chỗ ở nói chung như một nơi chốn ("chỗ ở dễ tìm", "chỗ ở đẹp"); sự thuận tiện đi lại, khoảng cách tới biển / chợ / trung tâm, view và cảnh nhìn từ chỗ ở | "đi bộ 2 phút là ra biển" → `vi_tri:pos` |
| `phong` | Phòng ở | độ rộng, độ thoáng, độ mới, vẻ đẹp của phòng; giường, nệm, gối (êm, cứng…); thiết bị trong phòng (điều hoà, máy lạnh, bình nóng lạnh, vòi sen) | "phòng rộng", "điều hoà hỏng" → `phong` |
| `ve_sinh` | Vệ sinh | sạch / bẩn, gọn gàng, mùi, ẩm, côn trùng, tóc, vết ố — của phòng, nhà vệ sinh, ga gối, khăn tắm; việc dọn phòng, thay ga gối | "phòng sạch" → `ve_sinh:pos`; "thấy cả tóc trên gối" → `ve_sinh:neg` |
| `phuc_vu` | Phục vụ | nhân viên, lễ tân, chủ nhà; dịch vụ, thủ tục nhận phòng / check-in | "lễ tân niềm nở" → `phuc_vu:pos` |
| `gia_ca` | Giá cả | giá phòng, phụ thu, các loại phí (kể cả "phí dịch vụ"), có đáng tiền không | "2 triệu một đêm mà như nhà nghỉ" → `gia_ca:neg` |
| `an_uong` | Ăn uống | bữa sáng, đồ ăn, đồ uống, cà phê, trà | "buffet sáng nghèo nàn" → `an_uong:neg` |
| `tien_ich` | Tiện ích chung | hồ bơi, phòng gym, sân vườn, khu vui chơi trẻ em, wifi / mạng, bãi đỗ xe, thang máy, máy giặt chung, xe đưa đón | "wifi chập chờn" → `tien_ich:neg` |
| `yen_tinh` | Độ yên tĩnh | tiếng ồn, khả năng cách âm (của phòng hay của khu vực xung quanh) | "phòng ồn", "nằm nghe karaoke tới 1 giờ sáng" → `yen_tinh:neg` |

Quy ước gán nhãn:

- Mỗi đánh giá có từ 1 đến 5 khía cạnh được đánh giá; mỗi khía cạnh có **nhiều nhất một** cực tính (không đánh giá nào vừa
  khen vừa chê cùng một khía cạnh). Một khía cạnh có thể được nói tới nhiều lần với cùng cực tính.
- Khía cạnh được xác định theo **nội dung được đánh giá**, không theo danh từ đứng đầu câu: "phòng sạch" là `ve_sinh`,
  "phòng rộng" là `phong`, "phòng ồn" là `yen_tinh`.
- Cực tính theo nghĩa của cả cụm: "không sạch", "không được sạch lắm", "chẳng sạch sẽ gì" → chê; "không hề tệ",
  "không có gì để chê" → khen; so sánh "sạch hơn hẳn chỗ lần trước" → khen.
- Đánh giá có thể không chứa từ khoá của khía cạnh ("đi bộ 2 phút là ra biển", "nằm nghe karaoke tới 1 giờ sáng").
- Câu **chỉ nhắc tới** một khía cạnh mà không khen hay chê thì **không** gán nhãn: "mình đặt phòng qua app",
  "lễ tân hỏi có cần gọi taxi không", "khách sạn có hồ bơi trên sân thượng", "giá phòng là 500k một đêm". Câu kể chuyện,
  lời khuyên, nhận xét chung ("nhìn chung rất hài lòng", "sẽ quay lại", "thất vọng") cũng không thuộc khía cạnh nào.
- `so_sao` là cảm nhận **chung** của khách về cả kỳ lưu trú, không phải điểm của từng khía cạnh.

## Dữ liệu

| Thư mục | Số đánh giá | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 16 000 | ✅ cột `labels` trong `train.csv` | huấn luyện |
| `data/test_public/` | 800 | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm cục bộ |
| tập chấm ẩn | 3 000 | ❌ | chấm chính thức — cùng định dạng và kiểu phân phối với `test_public` |

```
data/train/train.csv                    # id, text, so_sao, nen_tang, labels
data/test_public/test.csv               # id, text, so_sao, nen_tang
data/test_public/answers.csv            # id, labels, khong_dau, hon_hop — bản sao dev, KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv  # id, labels
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`. Tập chấm ẩn do BTC giữ, cùng định dạng `test_public/test.csv`
(id dạng `prv_000000`), không có đáp án; lúc chấm nó thay thế `data/test_public/`.

| Cột | Ý nghĩa |
|---|---|
| `id` | `tr_000000`, `pub_000000`… |
| `text` | nội dung đánh giá (UTF-8, một dòng, có thể chứa dấu phẩy — đọc bằng trình đọc CSV chuẩn) |
| `so_sao` | số sao khách cho, 1–5 |
| `nen_tang` | ứng dụng đặt phòng: `vivu`, `dimuon`, `nghiduong` (train và test), `phuot` (**chỉ có ở test**) |
| `labels` | các cặp `khia_canh:cuc_tinh` cách nhau bởi một dấu cách, ví dụ `phong:pos ve_sinh:neg` |
| `khong_dau` (chỉ trong `answers.csv`) | 1 nếu đánh giá được gõ không dấu / viết tắt / teencode / sai chính tả, 0 nếu gõ chuẩn |
| `hon_hop` (chỉ trong `answers.csv`) | 1 nếu đánh giá vừa có khía cạnh được khen vừa có khía cạnh bị chê |

**Lưu ý phân phối.** `train` lấy từ ba ứng dụng cũ: phần lớn gõ chuẩn có dấu và phần lớn là đánh giá **một chiều** (khen
tất cả hoặc chê tất cả). Ở `test_public` và tập chấm ẩn: gần một nửa đánh giá đến từ `phuot`; khoảng một nửa đánh giá gõ
không chuẩn — không dấu, viết tắt / teencode (`ko`, `k`, `dc`, `nv`, `ks`, `p`, `j`, `wa`…), kéo dài chữ ("đẹppp"), sai chính tả,
có khi bỏ cả dấu câu; khoảng một nửa đánh giá **vừa khen vừa chê**. Một số cách diễn đạt chỉ xuất hiện ở test: những mảnh quen
thuộc ghép theo kiểu mới (một đối tượng đi với một tính từ chưa từng đi cùng nhau trong train, một tính từ quen thuộc trong cấu
trúc phủ định / so sánh chưa gặp với nó); tập chấm ẩn có những cách ghép mới **khác** với `test_public`. Từ ngữ dùng để khen chê
ở test đều đã xuất hiện trong train (chỉ có thể khác cách viết). `test_public` phản ánh đúng các dịch chuyển này. Không có đánh
giá nào ở test trùng nguyên văn với train.

## Định dạng nộp

File CSV `submission.csv`, header `id,labels`, mỗi đánh giá của tập test đúng một dòng:

```csv
id,labels
pub_000000,phong:pos ve_sinh:neg
pub_000001,gia_ca:neg
pub_000002,
```

- `labels`: các cặp `khia_canh:cuc_tinh` cách nhau bởi dấu cách, `khia_canh` thuộc 8 mã ở trên, `cuc_tinh` là `pos` hoặc `neg`;
  thứ tự không quan trọng; **chuỗi rỗng hợp lệ** (dự đoán không có khía cạnh nào).
- Thiếu / trùng / thừa `id`, nhãn không hợp lệ, hoặc một khía cạnh xuất hiện hai lần trong cùng một dòng → bài nộp **bị từ chối**
  (score.py báo lỗi).

## Ràng buộc

- Chỉ dùng dữ liệu BTC cấp. Không dữ liệu ngoài, không mô hình / từ điển / bộ chuẩn hoá tải từ internet, không LLM / API.
  Được tự viết luật chuẩn hoá (bỏ dấu, quy đổi viết tắt…) dựa trên quan sát dữ liệu.
- **Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn tham số** — chỉ để tự chấm. Được dùng văn bản test
  **không nhãn** (của `test_public` và của tập chấm ẩn) theo bất kỳ cách nào: chuẩn hoá, học không giám sát, tự gán nhãn…
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 10 phút trên môi trường chấm.
- Nộp kèm code và báo cáo kỹ thuật ngắn.

## Chấm điểm

Gọi $T$ là tập các bộ ba (đánh giá, khía cạnh, cực tính) trong đáp án và $\hat T$ là tập bộ ba trong bài nộp, trên toàn bộ
tập chấm. Điểm thô là **micro-F1**:

$$\text{micro-F1} = \frac{2\,|\hat T \cap T|}{|\hat T| + |T|}$$

- Đoán đúng khía cạnh nhưng sai cực tính bị tính là **một bộ thừa và một bộ thiếu**. Bỏ trống một đánh giá không bị phạt thêm
  ngoài các bộ ba bị thiếu.
- `score.py` báo thêm: `aspect_f1` (micro-F1 chỉ xét khía cạnh, bỏ cực tính), `polarity_acc` (tỉ lệ đúng cực tính trên các khía cạnh
  phát hiện đúng), `f1_khong_dau` và `f1_hon_hop` (micro-F1 trên nhóm đánh giá có `khong_dau = 1` / `hon_hop = 1`).

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{micro-F1} - 0.725}{0.97 - 0.725},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py`: TF-IDF theo từ (1–2 gram) trên **văn bản gốc** + One-vs-Rest Logistic Regression cho 16 nhãn
`khia_canh:cuc_tinh`, ngưỡng 0.5; nếu cả hai cực tính của một khía cạnh vượt ngưỡng thì giữ cực tính có xác suất cao hơn.
Điểm **micro-F1 = 0.725** trên tập ẩn (≈ 0 điểm; 0.736 trên `test_public`). Điểm BTC **0.97** đạt được bằng mô hình cổ điển
(scikit-learn, CPU, khoảng 1–2 phút) trong giới hạn thời gian; tối đa của metric là 1.0.

```
python baseline.py                     # đọc data/, ghi submission.csv
python score.py --pred submission.csv  # chấm trên test_public
```
