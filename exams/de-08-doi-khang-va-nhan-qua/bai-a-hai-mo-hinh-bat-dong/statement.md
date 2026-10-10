# Bài A — Hai giám định viên

- **Thời gian chạy:** 20 phút (wall-clock cho toàn bộ quá trình tạo nhiễu trên tập chấm ẩn lúc chấm, kể cả mọi bước chuẩn bị)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (chỉ dùng hai mô hình BTC cấp trong `data/models.npz` — đó là đối tượng cần đánh lừa)
- **Điểm baseline:** score = 0.04
- **Điểm BTC (100 điểm):** score = 0.95

## Bối cảnh

Một công ty khởi nghiệp ở Đà Nẵng làm hệ thống hỗ trợ lái xe nhận dạng biển báo giao thông từ camera hành trình.
Để "chắc ăn", mỗi biển báo được hai giám định viên AI cùng xem: **M** — một mạng nơ-ron nhìn tổng thể mọi điểm ảnh,
và **F** — một rừng cây quyết định soi từng điểm ảnh theo ngưỡng. Trên mọi ảnh kiểm thử, cả hai đều trả lời đúng, và
hệ thống chỉ phát cảnh báo khi hai giám định viên **bất đồng**. Trung tâm An toàn AI được mời kiểm định độ "khác nhau
trong suy nghĩ" của hai mô hình: liệu có thể chỉ thay đổi rất nhẹ một bức ảnh — mắt thường khó nhận ra — để một giám
định viên nhầm còn người kia vẫn đúng, theo đúng chiều mình muốn?

Trưởng nhóm kiểm định dặn: *"Làm một người nhầm thì dễ — cái khó là người kia vẫn phải nói đúng. Hai giám định viên
học từ cùng một bộ ảnh, nên một vết nhiễu đủ mạnh thường lừa được cả hai. Ông F không có đạo hàm để lần theo, và nhiễu
lung tung thì ông ấy chẳng mảy may — nhưng ông ấy có điểm yếu riêng. À, biên bản nộp lưu bằng float16 và ảnh bị cắt về
[0, 1]: thứ gì nằm sát ngưỡng quá có thể trượt sang phía bên kia."*

## Nhiệm vụ

Mỗi mẫu là một ảnh biển báo giao thông Việt Nam **tổng hợp**, RGB 32×32, kèm **nhãn đúng** $y$ (12 lớp, bảng dưới).
Nhãn đúng được cho sẵn ở mọi tập, kể cả tập chấm ẩn. Gọi $x \in [0,1]^{32\times 32\times 3}$ là ảnh (giá trị pixel PNG chia 255).
Với mỗi ảnh, bạn tạo **hai** nhiễu $\delta_A, \delta_B$ cùng kích thước 32×32×3:

- **Loại A:** ảnh sau khi thêm $\delta_A$ làm **M sai** nhưng **F vẫn đúng**.
- **Loại B:** ảnh sau khi thêm $\delta_B$ làm **F sai** nhưng **M vẫn đúng**.

Ràng buộc: $\|\delta\|_\infty \le \varepsilon = 16/255$ (mọi phần tử). Ảnh đưa vào hai mô hình là
$x' = \mathrm{clip}(x + \delta,\ 0,\ 1)$, trong đó $\delta$ được ép về float32 trước khi cộng. Nhiễu thành công càng nhỏ
(theo chuẩn $L_2$) càng được nhiều điểm.

Một mô hình **đúng** trên $x'$ khi điểm của lớp $y$ **lớn hơn hẳn** điểm của mọi lớp khác (logit với M, xác suất bỏ phiếu
với F); **hoà được tính là sai**. Hàm `evaluate` trong `models.py` cài đặt chính xác quy tắc mà `score.py` dùng để chấm.

| Nhãn | Mã lớp | Biển báo |
|---|---|---|
| 0 | `cam_nguoc_chieu` | Cấm đi ngược chiều |
| 1 | `cam_re_trai` | Cấm rẽ trái |
| 2 | `toc_do_40` | Tốc độ tối đa cho phép 40 km/h |
| 3 | `toc_do_50` | Tốc độ tối đa cho phép 50 km/h |
| 4 | `toc_do_60` | Tốc độ tối đa cho phép 60 km/h |
| 5 | `duong_uu_tien` | Đường ưu tiên |
| 6 | `nguoi_di_bo` | Đường người đi bộ cắt ngang |
| 7 | `tre_em` | Trẻ em |
| 8 | `cong_truong` | Công trường |
| 9 | `dung_lai` | Dừng lại (STOP) |
| 10 | `cam_do_xe` | Cấm đỗ xe |
| 11 | `cam_dung_do_xe` | Cấm dừng xe và đỗ xe |

## Dữ liệu

| Thư mục / file | Số ảnh | Có nhãn? | Dùng để |
|---|---|---|---|
| `data/models.npz` | — | — | trọng số hai mô hình M và F (đối tượng cần đánh lừa) |
| `data/train/` | 4 800 (12 lớp × 400) | ✅ `labels.csv` | dữ liệu BTC đã dùng để huấn luyện M và F; được dùng tuỳ ý (ví dụ huấn luyện mô hình phụ trợ) |
| `data/test_public/` | 204 (12 lớp × 17) | ✅ `labels.csv` | thử nghiệm và tự chấm cục bộ |
| tập chấm ẩn | 204 (12 lớp × 17) | ✅ `labels.csv` | chấm chính thức — ảnh mới hoàn toàn, cùng định dạng và phân phối với `test_public` |

```
data/models.npz
data/train/images/tr_000000.png ...          # ảnh RGB 32x32
data/train/labels.csv                        # id,label
data/test_public/images/pub_000000.png ...
data/test_public/labels.csv                  # id,label  (nhãn đúng — là dữ liệu vào của bài)
data/test_public/sample_submission.npz       # bài nộp mẫu (toàn số 0)
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.

Mọi ảnh trong `test_public` và tập chấm ẩn đều được **cả hai** mô hình phân loại đúng (theo quy tắc "hơn hẳn" ở trên),
rừng F bỏ phiếu cho lớp đúng với cách biệt ít nhất 0.1, và BTC đã loại những ảnh mà về lý thuyết không thể hoặc gần như
không thể làm F sai trong ngân sách $\varepsilon$ (cận trên của hiệu xác suất giữa lớp mạnh nhất còn lại và lớp đúng ≤ 0.05).
Tập chấm ẩn do BTC giữ, cùng định dạng `test_public`; lúc chấm nó thay thế `data/test_public/`,
nên code phải đọc danh sách ảnh từ `labels.csv` của thư mục đó.

### Hai mô hình

Ảnh được làm phẳng thành vector $v \in \mathbb{R}^{3072}$ theo thứ tự hàng → cột → kênh (pixel hàng $r$, cột $c$, kênh $k$ có
chỉ số $(32r + c)\cdot 3 + k$), tức `x.reshape(N, -1)`.

- **M (MLP):** $z = (v - \mu)/\sigma$, $\;h = \mathrm{ReLU}(z W_1 + b_1)$ (128 nơ-ron), $\;\text{logit} = h W_2 + b_2$ (12 lớp).
- **F (rừng ngẫu nhiên):** 100 cây, độ sâu tối đa 14. Tại mỗi nút trong, đi sang **trái** nếu
  $v[\text{feature}] \le \text{threshold}$ (so sánh float32), ngược lại sang phải. Mỗi lá có một phân phối lớp;
  xác suất của F là **trung bình** phân phối lá của 100 cây.

| Mảng trong `models.npz` | Kích thước, kiểu | Ý nghĩa |
|---|---|---|
| `mlp_mean`, `mlp_scale` | (3072,) float32 | $\mu$, $\sigma$ chuẩn hoá đầu vào của M |
| `mlp_W1`, `mlp_b1` | (3072, 128), (128,) float32 | lớp ẩn |
| `mlp_W2`, `mlp_b2` | (128, 12), (12,) float32 | lớp ra |
| `forest_feature` | (92 408,) int16 | chỉ số pixel được so sánh tại nút; −1 nếu là lá |
| `forest_threshold` | (92 408,) float32 | ngưỡng tại nút |
| `forest_left`, `forest_right` | (92 408,) int32 | chỉ số (toàn cục) nút con trái/phải; −1 nếu là lá |
| `forest_value` | (92 408, 12) float32 | phân phối lớp tại lá (bằng 0 ở nút trong) |
| `forest_roots` | (100,) int32 | chỉ số nút gốc của từng cây |
| `forest_max_depth` | () int32 | độ sâu lớn nhất của các cây |

File `models.py` (đi kèm đề, chỉ dùng numpy) cài sẵn: `load_models`, `to_float`, `apply_delta`, `mlp_logits`,
`forest_leaves`, `forest_proba`, `is_correct`, `penalty_factor`, `evaluate`. `score.py` chấm bằng chính các hàm này.
Không có hàm tính gradient — bạn tự cài nếu cần.

## Định dạng nộp

Một file `.npz` (tạo bằng `np.savez` hoặc `np.savez_compressed`, không pickle) gồm ba mảng sau (mảng thừa nếu có sẽ bị bỏ qua):

| Mảng | Kích thước | Kiểu |
|---|---|---|
| `ids` | (N,) | chuỗi, ví dụ `pub_000000` |
| `delta_a` | (N, 32, 32, 3) | float16 (khuyến nghị) hoặc float32 |
| `delta_b` | (N, 32, 32, 3) | float16 (khuyến nghị) hoặc float32 |

```python
with open("submission.npz", "wb") as f:
    np.savez_compressed(f, ids=np.array(ids), delta_a=delta_a.astype(np.float16), delta_b=delta_b.astype(np.float16))
```

- Mỗi id của `labels.csv` xuất hiện đúng một lần (thứ tự tuỳ ý); mọi phần tử hữu hạn và $|\delta| \le 16/255$
  (dung sai $10^{-4}$ cho sai số làm tròn). Thiếu/trùng/thừa id, sai kích thước, sai kiểu, NaN/vô cực hoặc vượt ngân sách
  → bài nộp **bị từ chối**.
- Muốn bỏ trống một nhiễu, nộp mảng toàn 0 cho ảnh đó (nhiễu 0 không bao giờ thành công vì cả hai mô hình đều đúng trên ảnh gốc).
- Kiểm tra lại bài nộp **sau khi** ép kiểu float16 bằng `models.evaluate` — đó chính là phép chấm.

## Ràng buộc

- Chỉ dùng dữ liệu và hai mô hình BTC cấp. Không dữ liệu ngoài, không trọng số pretrained, không LLM/API.
- Không được thay đổi `models.npz` hay quy tắc chấm; được tự do truy vấn hai mô hình, đọc trọng số và cấu trúc cây.
- Code phải **tự động** tạo nhiễu cho ảnh mới (tập chấm ẩn gồm ảnh khác hoàn toàn `test_public`), không chỉnh tay từng ảnh.
  `test_public` chỉ dùng để thử nghiệm và tự chấm.
- Toàn bộ pipeline chạy xong trong 20 phút trên môi trường chấm. Nộp kèm code và báo cáo kỹ thuật ngắn.
- Mẹo vận hành: nếu chạy nhiều tiến trình song song (ví dụ chia ảnh cho 4 tiến trình), hãy đặt `OMP_NUM_THREADS=1`
  để các tiến trình không tranh luồng BLAS — trên máy đang tải, bỏ qua bước này có thể làm chậm nhiều lần.

## Chấm điểm

Với mỗi nhiễu, hệ số phạt theo độ lớn:

$$\mathrm{PF}(\delta) = 1 - 0.5\cdot\min\left(1,\ \frac{\|\delta\|_2}{L}\right), \qquad L = \frac{16}{255}\sqrt{3072} \approx 3.478$$

($\|\delta\|_2$ tính trên nhiễu **đã nộp**, kể cả phần bị cắt mất khi clip). Với ảnh $i$, gọi $A_i$ / $B_i$ là sự kiện
nhiễu loại A / loại B thành công:

$$s_i = \frac12\Big(\mathbb{1}[A_i]\cdot \mathrm{PF}(\delta_{A,i}) + \mathbb{1}[B_i]\cdot \mathrm{PF}(\delta_{B,i})\Big), \qquad
\text{score} = \frac1N\sum_{i=1}^{N} s_i$$

Giá trị score (trong [0, 1]) là **điểm thô** của bài; $N = 204$ ở tập chấm ẩn.

Hành vi của metric:
- Nhiễu thất bại được 0 bất kể lớn nhỏ; nhiễu thành công được từ 0.5 đến 1. Nhiễu dùng hết ngân sách $\pm\varepsilon$ ở mọi
  phần tử có $\|\delta\|_2 = L$ nên PF = 0.5; nhiễu thành công với $\|\delta\|_2 = 0.35$ có PF ≈ 0.95.
- Hai loại nặng ngang nhau: làm tốt một loại, bỏ trống loại kia thì score không quá 0.5.
- `score.py` báo thêm tỉ lệ thành công từng loại (`success_a`, `success_b`) và PF trung bình của các nhiễu thành công.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{score} - 0.04}{0.95 - 0.04},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.npz` trên `test_public`; đổi file bằng `--pred`).

## Baseline

`baseline.py`: với mỗi ảnh và mỗi loại, thử tối đa 10 nhiễu dấu ngẫu nhiên $\pm 16/255$ trên mọi phần tử, giữ nhiễu đầu tiên
đạt yêu cầu (kiểm bằng `models.evaluate`), không có thì nộp nhiễu 0. Nhiễu ngẫu nhiên hiếm khi làm hai giám định viên
bất đồng: chỉ ≈ 5% ảnh thành công loại A và ≈ 9% loại B, mỗi lần thành công chỉ được PF = 0.5. Điểm: **score = 0.04**
trên tập ẩn. Điểm BTC **0.95** đạt được bằng một lời giải chỉ dùng numpy, chạy vài phút trên CPU: thành công gần như
100% ở cả hai loại với PF trung bình ≈ 0.95. Tối đa của metric là 1.0.
