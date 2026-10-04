# Bài A — Tuần lễ của điều độ viên

- **Thời gian chạy:** 10 phút (toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** CPU 4 nhân, 16 GB RAM, không có internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** pinball loss chuẩn hoá = 0.141 (trên tập chấm ẩn)
- **Điểm BTC (100 điểm):** pinball loss chuẩn hoá = 0.045

## Bối cảnh

Trung tâm Điều độ hệ thống điện phải lập kế hoạch huy động nguồn cho tuần tới. Một dự báo "đúng
trung bình" là chưa đủ: điều độ viên cần biết **khoảng bất định** — tải có thể thấp tới đâu (P10) và
cao tới đâu (P90) — để quyết định dự phòng quay. Tuần cần dự báo lại rơi đúng vào **kỳ nghỉ 30/4–1/5**,
khi các khu công nghiệp ngừng máy, văn phòng đóng cửa, còn khu dân cư thì bật điều hoà cả ngày giữa
nắng nóng đầu hè. Bạn được giao số liệu đo đếm hằng giờ của ~40 trạm biến áp 110 kV ở ba miền.

## Nhiệm vụ

Với mỗi trạm và mỗi giờ trong **168 giờ (7 ngày) kế tiếp** kể từ mốc dự báo, dự đoán ba phân vị của
phụ tải (MW): `p10`, `p50`, `p90`.

Những gì làm bài toán khó:

- **Mùa vụ ngày/tuần** khác nhau giữa các trạm (cơ cấu dân dụng / thương mại / công nghiệp không được
  cho biết — phải tự suy ra từ hình dạng phụ tải); phản ứng với ngày lễ cũng khác nhau giữa các trạm.
- **Nhiệt độ**: phụ tải tăng phi tuyến khi trời nóng (miền Bắc còn tăng khi rét), có quán tính nhiệt,
  và ngày nghỉ ban ngày nhạy với nhiệt độ hơn ngày làm việc. Cho kỳ dự báo bạn chỉ có **dự báo nhiệt
  độ** (sai số tăng theo lead time), không có nhiệt độ thực.
- **Ngày lễ Việt Nam**: lịch nghỉ (kể cả Tết âm lịch dịch chuyển hằng năm và ngày **làm bù**) có trong
  `holidays.csv`. Hành vi quanh Tết (trước/sau kỳ nghỉ) **không** được ghi trong lịch.
- **Sự cố**: lịch sử có các đoạn mất điện (giá trị `0`), mất số liệu (ô trống) và vài gai đo sai.
  Đáp án là phụ tải thực (không có sự cố).
- **Trạm mới (cold-start)**: vài trạm mới đóng điện chỉ có vài ngày đến vài tuần lịch sử.
- **Xu hướng & điểm gãy**: tăng trưởng khác nhau giữa các trạm; có trạm bị chuyển tải / thêm khách hàng
  lớn làm mức tải nhảy bậc — kể cả chỉ vài ngày trước mốc dự báo.
- **Độ bất định khác nhau**: có trạm tải rất đều, có trạm dao động mạnh — một dải ±x% cố định cho mọi
  trạm sẽ không hiệu chuẩn.

## Dữ liệu

Giải nén tại thư mục bài: `unzip public_data.zip` (tạo thư mục `data/`).

| Thư mục | Nội dung | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | phụ tải 40 trạm 2023-01-01 → 2024-04-24 (≈ 415 nghìn dòng), nhiệt độ thực 3 miền, lịch lễ 2023–2025, danh sách trạm | ✅ (chính là `load.csv`) | huấn luyện |
| `data/test_public/` | 168 h từ **2024-04-25 00:00** (Thứ Năm): 2 ngày làm việc rồi nghỉ liền 27/4–1/5/2024; 40 trạm × 168 h = 6 720 dòng | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm |

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` nhưng không có `answers.csv`. Lúc chấm,
`data/test_public/` được thay bằng tập ẩn: một tuần 168 h **muộn hơn, cùng kiểu** (2 ngày làm việc
rồi kỳ nghỉ 30/4–1/5 nhiều ngày liền), kèm:

- `history.csv` chứa số liệu đo **từ sau train tới mốc dự báo** (khoảng một năm, cũng có sự cố/thiếu số liệu);
- `weather.csv` có nhiệt độ thực của giai đoạn history (`kind=actual`) và dự báo cho 168 h (`kind=forecast`);
- có thể có **trạm mới** chưa xuất hiện trong train (lịch sử chỉ nằm trong `history.csv`, thông tin trạm ở `stations.csv` của tập đó).

Ở `test_public`, `history.csv` rỗng vì mốc dự báo trùng điểm kết thúc train. Pipeline của bạn
**phải** đọc lịch sử = `train/load.csv` + `<tập test>/history.csv`, danh sách trạm từ
`<tập test>/stations.csv`, và lấy mốc dự báo = thời điểm nhỏ nhất trong `<tập test>/test.csv`.
Không được hard-code ngày hay danh sách trạm.

```
data/train/load.csv            # station_id,timestamp,load_mw     (trống = mất số liệu, 0 = mất điện)
data/train/weather.csv         # region,timestamp,temp_c          (nhiệt độ thực, region ∈ {Bac, Trung, Nam})
data/train/stations.csv        # station_id,region,first_date     (ngày bắt đầu có số liệu)
data/train/holidays.csv        # date,name,type                   (type ∈ {tet, national, makeup_workday})
data/test_public/stations.csv  # các trạm cần dự báo
data/test_public/history.csv   # station_id,timestamp,load_mw     số liệu sau train (ở test_public: rỗng)
data/test_public/weather.csv   # region,timestamp,temp_c,kind     kind ∈ {actual, forecast}
data/test_public/test.csv      # station_id,timestamp             các dòng cần dự báo
data/test_public/answers.csv   # station_id,timestamp,load_mw     — chỉ có ở bản sao dev
```

`timestamp` theo giờ địa phương, định dạng `YYYY-MM-DD HH:MM`; giá trị là phụ tải trung bình của giờ đó.

## Định dạng nộp

Code ghi `submission.csv`, đúng một dòng cho mỗi dòng của `test.csv`:

```csv
station_id,timestamp,p10,p50,p90
TBA_101,2024-04-25 00:00,11.82,12.95,14.10
TBA_101,2024-04-25 01:00,11.05,12.20,13.31
```

- Thiếu dòng, trùng khoá `(station_id, timestamp)`, dòng lạ, giá trị trống/NaN/vô hạn → bài nộp bị từ
  chối (score.py báo lỗi, exit code 1).
- Nếu các phân vị "chéo nhau" (vd `p10 > p50`), ba giá trị được sắp xếp lại theo từng dòng trước khi chấm.

## Ràng buộc

- CPU, không internet, 10 phút cho cả huấn luyện + suy luận trên tập ẩn.
- Chỉ dùng dữ liệu BTC cấp; **không** dùng dữ liệu thời tiết/lịch bên ngoài, không trọng số pretrained.
- **Không** dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn tham số bằng cách khớp đáp án
  (chỉ để tự chấm). Hãy tự tạo các "mốc dự báo giả" trong train để kiểm định (backtest).

## Chấm điểm

Pinball loss của phân vị $q$ với giá trị thật $y$ và dự báo $\hat y_q$:

$$\rho_q(y,\hat y_q) = \max\big(q\,(y-\hat y_q),\ (q-1)\,(y-\hat y_q)\big)$$

Với mỗi trạm $s$ có tập giờ $T_s$ (168 giờ), loss được **chuẩn hoá theo quy mô trạm** $\bar y_s$ (tải
trung bình thật của trạm trong kỳ test), rồi lấy trung bình trên các trạm:

$$L_s = \frac{1}{\bar y_s}\cdot\frac{1}{3|T_s|}\sum_{t\in T_s}\ \sum_{q\in\{0.1,\,0.5,\,0.9\}} \rho_q\big(y_{s,t},\hat y_{s,t,q}\big),
\qquad \text{metric} = \frac{1}{|S|}\sum_{s\in S} L_s$$

- Càng nhỏ càng tốt. Mọi trạm có trọng số như nhau — trạm 6 MW quan trọng ngang trạm 120 MW, nên trạm
  mới / trạm nhỏ không thể bỏ qua.
- Pinball loss thưởng cả **độ chính xác** (thành phần P50 tương đương MAE/2) lẫn **hiệu chuẩn khoảng**:
  khoảng quá hẹp bị phạt nặng khi thực tế rơi ra ngoài, quá rộng bị phạt đều đặn.
- score.py in thêm `coverage_p10_p90` (tỉ lệ giá trị thật nằm trong [P10, P90], lý tưởng ≈ 0.8) để tham khảo.

Quy đổi ra thang 0–100 (metric giảm là tốt):

$$\text{điểm} = 100\cdot\operatorname{clip}\!\left(\frac{0.141 - \text{metric}}{0.141 - 0.045},\,0,\,1\right)$$

## Baseline

`baseline.py`: **seasonal naive** — P50 = phụ tải cùng giờ tuần trước; nếu giờ đó bị mất điện/thiếu số
liệu thì lấy cùng giờ hai tuần trước; trạm quá mới thì dùng trung vị theo giờ-trong-ngày. P10 = 0.85·P50,
P90 = 1.15·P50. Điểm thô ≈ **0.141** trên tập ẩn (≈ 0.154 trên `test_public`) — 0 điểm. Baseline chép
nguyên tuần trước nên dính các bẫy ngày lễ, ngày làm bù, điểm gãy, nhiệt độ và dải cố định.

```
python baseline.py                     # đọc data/, ghi submission.csv
python score.py --pred submission.csv  # chấm trên test_public
```
