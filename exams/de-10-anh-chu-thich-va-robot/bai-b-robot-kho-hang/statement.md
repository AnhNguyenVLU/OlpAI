# Bài B — Sáu chú robot kho hàng

- **Thời gian chạy:** 10 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** độ chính xác trung bình theo robot = 0.604
- **Điểm BTC (100 điểm):** độ chính xác trung bình theo robot = 0.96

## Bối cảnh

Một trung tâm phân phối ở Bình Dương vừa mua lại sáu robot kho hàng cũ từ ba nhà cung cấp khác nhau. Robot nhận lệnh
bằng tiếng Việt từ quản lý ca ("Đưa thùng đỏ ra cửa giao hàng nhé", "Lại gần xe đẩy hàng màu ghi") rồi tự xoay
xở trong kho. Để lập kế hoạch ca làm việc, đội vận hành cần một mô hình **đoán trước bước tiếp theo** của từng robot
chỉ từ một ảnh chụp trạng thái kho.

Kỹ sư bảo trì kể: *"Mỗi con một tật. Có con đi đường nào cũng phải xoay một kiểu, có con cứ phải đứng đúng tư thế mới
chịu gắp hàng, có con nghe lệnh xong thì chỉ nhớ mỗi cái màu. Nhưng tật của con nào thì con đó giữ mãi — xem kỹ nhật
ký của từng con là ra."*

## Nhiệm vụ

Mỗi mẫu là **một ảnh chụp độc lập** (không có lịch sử) tại một thời điểm trong lúc một robot thực hiện nhiệm vụ.
Dự đoán **hành động tiếp theo** của robot:

| mã | hành động | tác dụng |
|---:|---|---|
| 0 | tiến | đi lên ô phía trước nếu ô đó là sàn trống; nếu không thì đứng nguyên (va vào vật cản) |
| 1 | quay trái | xoay 90° ngược chiều kim đồng hồ tại chỗ |
| 2 | quay phải | xoay 90° theo chiều kim đồng hồ tại chỗ |
| 3 | nhặt | nhấc vật ở ô phía trước nếu đó là thùng hàng hoặc bao hàng và robot đang không mang gì |
| 4 | thả | đặt vật đang mang xuống ô sàn trống phía trước; nếu phía trước là cửa giao hàng thì vật được giao đi (biến mất) |
| 5 | đứng yên | robot cho rằng nhiệm vụ đã hoàn thành |

Hành động là **tương đối** với hướng robot đang quay mặt. Có bốn loại nhiệm vụ (mỗi loại nhiều cách diễn đạt):

| Loại | Ví dụ | Hoàn thành khi |
|---|---|---|
| đi tới X | "Lại gần xe đẩy hàng màu ghi", "Tiến về phía thùng hàng tím" | robot đứng ở ô kề X và quay mặt vào X |
| nhặt X | "Nhặt chiếc bao tải đỏ", "Nhấc hộp vàng lên giúp tôi" | robot đang mang X |
| đặt X cạnh Y | "Đặt thùng đỏ cạnh pallet vàng", "Xếp bao hàng xanh lá kế bên xe đẩy xám" | X nằm ở ô kề (trên/dưới/trái/phải) Y |
| mang X ra cửa giao hàng | "Mang thùng đỏ về cửa giao hàng", "Đem bao tím đến điểm giao hàng" | X đã được thả vào cửa giao hàng |

X, Y được nêu bằng loại vật + màu và là duy nhất trong kho (theo đúng loại + màu). Sáu robot (`robot` = 0…5) có
**thói quen riêng, ẩn nhưng nhất quán**: cách chọn đường, cách xoay, cách thao tác, thậm chí cách hiểu lệnh. Mỗi robot
cũng thỉnh thoảng (khoảng 3% số bước) làm một hành động ngẫu nhiên; nhãn là hành động robot **thực sự** đã làm.
Ảnh chụp lấy ở các thời điểm khác nhau của nhiệm vụ, kể cả lúc robot vừa xong việc (khi đó hành động là "đứng yên").
Mỗi ảnh chụp của `test_public` và của tập chấm ẩn thuộc về một lượt nhiệm vụ riêng (kho và lệnh khác nhau), nên các
hàng thật sự độc lập. Ở `train`, một lượt nhiệm vụ có thể cho tới 3 ảnh chụp (cùng kho, cùng lệnh) nằm lẫn trong file;
hãy cân nhắc điều này khi tự chia dữ liệu để kiểm định.

## Dữ liệu

| Thư mục | Số ảnh chụp | Có nhãn? | Dùng để |
|---|---|---|---|
| `data/train/` | 36 000 (6 000 mỗi robot) | ✅ cột `hanh_dong` | huấn luyện |
| `data/test_public/` | 3 000 (500 mỗi robot) | ✅ `answers.csv` (bản sao dev) | tự chấm cục bộ |
| tập chấm ẩn | 12 000 (2 000 mỗi robot) | ❌ | chấm chính thức — cùng định dạng và kiểu phân phối với `test_public` |

```
data/train/observations.csv           # id,robot,nhiem_vu,huong,mang_loai,mang_mau,luoi_loai,luoi_mau,hanh_dong
data/test_public/observations.csv     # như trên, không có hanh_dong
data/test_public/answers.csv          # id,robot,hanh_dong — bản sao dev, KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`. Tập chấm ẩn do BTC giữ, cùng định dạng
`test_public/observations.csv` (id dạng `prv_000000`), không có đáp án; lúc chấm nó thay thế `data/test_public/`.

| Cột | Ý nghĩa |
|---|---|
| `robot` | robot nào (0–5) |
| `nhiem_vu` | lệnh tiếng Việt |
| `huong` | hướng robot đang quay mặt: 0 = lên (hàng − 1), 1 = phải (cột + 1), 2 = xuống (hàng + 1), 3 = trái (cột − 1) |
| `mang_loai`, `mang_mau` | loại và màu của vật robot đang mang (0, 0 nếu không mang gì) |
| `luoi_loai`, `luoi_mau` | kho 9×9 (tường bao + 7×7 bên trong) dưới dạng chuỗi 81 chữ số, đọc theo hàng từ trên xuống, mỗi hàng từ trái sang phải: ô (hàng $r$, cột $c$) là ký tự thứ $9r + c$ |

Mã loại ô: 0 sàn trống, 1 tường, 2 kệ hàng (vật cản), 3 thùng hàng, 4 bao hàng, 5 pallet, 6 xe đẩy, 7 cửa giao hàng
(nằm trên tường bao), 8 robot. Mã màu: 0 không màu (sàn, tường, kệ, cửa, robot), 1 đỏ, 2 xanh lá, 3 xanh dương,
4 vàng, 5 tím, 6 xám. Robot chỉ đi trên sàn trống; thùng và bao nhặt được; pallet và xe đẩy là mốc cố định.

**Lưu ý phân phối.** Câu lệnh test dùng một số **tổ hợp** cách nói / loại vật / màu chưa từng xuất hiện trong train
(mọi từ đều đã có trong train); tập chấm ẩn có thể dùng những tổ hợp mới khác. Trạng thái kho của tập ẩn là các kho
và nhiệm vụ mới, cùng kiểu với `test_public`.

## Định dạng nộp

File CSV `submission.csv`, header `id,hanh_dong`, mỗi `id` của `observations.csv` đúng một dòng, `hanh_dong` là số
nguyên trong {0, 1, 2, 3, 4, 5}:

```csv
id,hanh_dong
pub_000000,0
pub_000001,2
```

Thiếu / trùng / thừa id, giá trị rỗng hoặc ngoài miền → bài nộp **bị từ chối**.

## Ràng buộc

- Chỉ dùng dữ liệu BTC cấp. Không dữ liệu ngoài, không mô hình pretrained, không LLM/API.
- **Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn mô hình** — chỉ để tự chấm.
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 10 phút trên môi trường chấm.
- Nộp kèm code và báo cáo kỹ thuật ngắn (mô tả những thói quen bạn phát hiện được ở từng robot, nếu có).

## Chấm điểm

Gọi $T_k$ là tập ảnh chụp của robot $k$ trong tập chấm. Điểm thô là **độ chính xác trung bình theo robot**:

$$\text{acc} = \frac{1}{6}\sum_{k=0}^{5} \frac{1}{|T_k|}\sum_{i \in T_k} \mathbb{1}\left[\hat a_i = a_i\right]$$

- Mỗi robot có trọng số như nhau. `score.py` báo thêm độ chính xác tổng và độ chính xác của từng robot.
- Do khoảng 3% hành động là ngẫu nhiên, không lời giải nào đạt 1.0; luôn đoán "tiến" chỉ được khoảng 0.25.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{acc} - 0.604}{0.96 - 0.604},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py`: LogisticRegression trên vài đặc trưng thô — robot, robot có đang mang vật không, loại và màu của ô ngay
phía trước robot — cộng túi từ của câu lệnh. Nó học được các quy tắc cục bộ (phía trước trống thì thường tiến, bị tường
hoặc kệ chặn thì quay, trước mặt là thùng hoặc bao thì nhặt hay đứng yên) nhưng không biết mục tiêu ở đâu nên không
chọn được hướng rẽ, cũng không phân biệt tật của từng robot. Điểm **0.604** trên tập ẩn. Điểm BTC **0.96** đạt được bằng
mô hình cổ điển (numpy/scikit-learn, CPU, vài phút); trần thực tế thấp hơn 1.0 vì nhiễu hành động.
