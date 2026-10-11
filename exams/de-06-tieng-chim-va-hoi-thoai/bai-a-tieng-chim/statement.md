# Bài A — Lắng nghe Cúc Phương

- **Thời gian chạy:** 20 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (mọi trọng số phải huấn luyện từ đầu trên `train/`)
- **Điểm baseline:** macro-AP = 0.19
- **Điểm BTC (100 điểm):** macro-AP = 0.78

## Bối cảnh

Vườn quốc gia Cúc Phương muốn theo dõi quần thể chim bằng các **máy ghi âm tự động** treo trên cây: mỗi máy
thu những đoạn "soundscape" dài 5 giây suốt ngày đêm. Không ai đủ sức nghe hết hàng nghìn giờ âm thanh, nên
ban quản lý cần một mô hình cho biết trong mỗi đoạn có **những loài nào** đang kêu.

Dữ liệu gán nhãn duy nhất là bộ sưu tập của các tình nguyện viên: những đoạn ghi **tiêu điểm** ngắn, cầm máy
hướng thẳng vào một con chim, mỗi đoạn chỉ ghi tên **một** loài chính. Một kiểm lâm lâu năm xem qua rồi dặn:
*"Trong rừng chẳng con chim nào hót một mình đâu, mưa gió, ve sầu, ếch nhái cũng chẳng chờ ai. Máy treo trên cây
nghe khác hẳn máy cầm tay. Còn nữa, đừng nhận chim qua cao độ: có những loài hót cùng một dải tần, chỉ khác nhau
ở nhịp và dáng nốt. Mà cùng một loài, mỗi con lại hót cao thấp, nhanh chậm một khác."*

Toàn bộ tiếng chim trong bài là **âm thanh tổng hợp** mô phỏng đặc điểm tiếng kêu (chuỗi âm tiết, quét tần số, nốt
có hoạ âm, láy, tiếng gõ...). Các đặc điểm này đã được đơn giản hoá, không trùng khớp với tiếng kêu thật của loài
mang tên tương ứng.

## Nhiệm vụ

Với mỗi đoạn soundscape 5 giây của tập test, dự đoán **xác suất có mặt** của từng loài trong 20 loài
(`train/species.csv`). Một đoạn có thể có 0, 1, 2 hoặc 3 loài kêu chồng lên nhau.

## Dữ liệu

| Thư mục | Số đoạn | Độ dài | Có đáp án? | Dùng để |
|---|---|---|---|---|
| `data/train/` | 871 | 1.6–2.6 s | ✅ `labels.csv` (một loài chính) | huấn luyện |
| `data/test_public/` | 300 | 5 s | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm cục bộ |
| tập chấm ẩn | 1 200 | 5 s | ❌ | chấm chính thức — cùng định dạng và phân phối với `test_public` |

```
data/train/audio/tr_000000.wav ...      # bản ghi tiêu điểm
data/train/labels.csv                   # id,species        (mã loài chính của đoạn ghi)
data/train/species.csv                  # code,ten_viet,ten_khoa_hoc   (20 loài)
data/test_public/audio/pub_000000.wav   # soundscape 5 s từ máy ghi tự động
data/test_public/answers.csv            # id + 20 cột 0/1 theo mã loài — bản sao dev, KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.

- Mọi file là WAV mono, tần số lấy mẫu **6 000 Hz**, PCM **8-bit không dấu** (đọc bằng `scipy.io.wavfile.read`,
  giá trị 0–255, im lặng = 128). Mỗi đoạn được chuẩn hoá biên độ đỉnh riêng.
- **train**: mỗi đoạn có một loài chính ở gần (33–56 đoạn mỗi loài). Khoảng **15%** đoạn có thêm một loài khác
  kêu nhỏ phía xa mà **không được gán nhãn**. Nền âm thanh tương đối yên tĩnh; ghi bằng nhiều máy cầm tay khác nhau.
- **test**: soundscape từ máy ghi tự động: 0–3 loài ở các khoảng cách khác nhau (càng xa càng nhỏ và càng mất
  âm cao), tỉ lệ tín hiệu/nhiễu thấp hơn hẳn, có mưa, gió, suối, côn trùng, ếch nhái; mỗi máy ghi có đáp ứng tần số
  riêng. Tập chấm ẩn được thu bằng **các máy ghi khác** với `test_public` (cùng kiểu thiết bị); tần suất các loài
  tương tự `test_public`.
- Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` nhưng **chỉ có `audio/`**; lúc chấm nó thay thế
  `data/test_public/`, nên code phải liệt kê file trong `audio/` thay vì đọc `answers.csv`.

## Định dạng nộp

File `submission.csv`: cột `id` và đúng 20 cột mang **mã loài** (như `train/species.csv`), mỗi đoạn test một dòng:

```csv
id,bim_bip,cu_gay,cu_meo,tu_hu,bat_co,cu_rot,khuou,chao_mao,hoa_mi,chich_choe,dieu_hau,vanh_khuyen,chim_sau,chich_bong,chia_voi,go_kien,chim_khach,sao,ga_rung,che_beo
pub_000000,0.02,0.01,0.00,0.05,0.10,0.01,0.03,0.80,0.12,0.02,0.01,0.00,0.04,0.01,0.02,0.00,0.01,0.03,0.00,0.01
```

- Mỗi `id` của tập test xuất hiện đúng một lần; mọi giá trị là số thực trong [0, 1] (không rỗng, NaN, inf, `True`/`False` hay chữ).
- Thiếu cột loài, thiếu/trùng/thừa id hoặc giá trị không hợp lệ → bài nộp **bị từ chối** (score.py báo lỗi).

## Ràng buộc

- Chỉ dùng dữ liệu trong `train/`. Không dữ liệu ngoài, không pretrained, không LLM/API.
- **Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn ngưỡng** (kể cả làm dữ liệu bổ sung, hiệu chỉnh hay stacking,
  **kể cả sao chép hoặc nhúng dữ liệu/trọng số lấy từ `test_public` vào mã nộp**) — chỉ để tự chấm.
  Được dùng âm thanh test (không nhãn) để chuẩn hoá hoặc thích nghi miền không giám sát lúc suy luận.
- Được tự tổng hợp nhiễu/biến đổi để tăng cường dữ liệu từ chính các đoạn `train/`.
- **Khi chấm, BTC chạy lại toàn bộ pipeline (huấn luyện + suy luận) từ mã nộp trong một thư mục chỉ có `data/train/` và tập chấm ẩn**
  (đặt ở `data/test_public/`, chỉ có `audio/`; không có `answers.csv`, `sample_submission.csv`). Vì vậy mã phải tự huấn luyện từ
  `train/`, không kèm trọng số hay dữ liệu tính sẵn. Mã chạy được bằng một lệnh duy nhất (ví dụ `python solution.py`; tên file do BTC quy định)
  và ghi `submission.csv` vào thư mục hiện tại; file dự đoán do BTC tạo khi chạy lại nên không cần nộp kèm. BTC rà soát mã của các đội đứng đầu.
- Toàn bộ pipeline huấn luyện + suy luận chạy xong trong 20 phút. Nộp kèm code và báo cáo kỹ thuật ngắn.

## Chấm điểm

Với mỗi loài $k$ có ít nhất một đoạn dương tính trong tập chấm, sắp xếp các đoạn theo xác suất dự đoán giảm dần
và tính **Average Precision**:

$$\mathrm{AP}_k = \sum_{n} \big(R_k(n) - R_k(n-1)\big)\, P_k(n), \qquad
\text{macro-AP} = \frac{1}{|\mathcal K|}\sum_{k \in \mathcal K} \mathrm{AP}_k$$

trong đó $P_k(n)$, $R_k(n)$ là precision và recall khi lấy $n$ đoạn có điểm cao nhất (như
`sklearn.metrics.average_precision_score`), $\mathcal K$ là tập loài có mẫu dương (cả 20 loài với tập chấm chuẩn).

- AP chỉ phụ thuộc **thứ tự** các đoạn theo từng loài: không cần hiệu chỉnh xác suất, nhưng điểm của mỗi loài
  phải so sánh được giữa các đoạn. Nộp hằng số cho AP ≈ tần suất loài (macro-AP ≈ 0.08).
- Mọi loài nặng ngang nhau, kể cả loài ít gặp.
- `score.py` báo thêm **micro-AP** (AP trên toàn bộ cặp đoạn × loài) để tham khảo.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{macro-AP} - 0.19}{0.78 - 0.19},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

`test_public` và tập chấm ẩn mỗi tập chỉ có 4 máy ghi, mỗi máy một đáp ứng tần số riêng, nên điểm tự chấm trên `test_public` có thể lệch so
với tập ẩn vài điểm: với các lời giải đã thử, `test_public` dễ hơn tập ẩn khoảng 0.01–0.05 điểm thô (2–8 điểm quy đổi, vì khoảng cách từ baseline
đến điểm BTC là 0.59). Đừng chỉ dựa vào `test_public` để đánh giá hay chọn mô hình.

## Baseline

`baseline.py`: với mỗi đoạn, tính phổ STFT, gộp thành 16 dải tần (chia đều theo thang log từ 150 Hz), lấy
**trung bình và độ lệch chuẩn theo thời gian** của log-năng lượng từng dải (32 đặc trưng), rồi huấn luyện
One-vs-Rest LogisticRegression trên train. Điểm: **macro-AP = 0.19** trên tập ẩn (≈ 0.23 trên `test_public`).
Điểm BTC **0.78** nằm giữa kết quả của một mạng tích chập nhỏ huấn luyện từ đầu trên log-mel (≈ 0.76–0.77) và của trung bình xác suất
hai mạng như vậy (≈ 0.81); cả hai đều tăng cường dữ liệu bằng cách trộn các đoạn `train/` thành soundscape giả (mỗi mạng cần cỡ 13 phút-CPU, tức vài phút
trên 4 CPU, nằm trong giới hạn thời gian; có GPU thì nhanh hơn nhưng lúc chấm chỉ có 4 CPU). Tối đa của metric là 1.0.
