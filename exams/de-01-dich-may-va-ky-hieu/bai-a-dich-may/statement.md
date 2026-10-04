# Bài A — Phiên dịch tiếng Kơ Ru

- **Thời gian chạy:** 10 phút (toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** CPU 4 nhân, 16 GB RAM (có thể có một GPU ≈16 GB), không có internet
- **Mô hình pretrained:** không được dùng (không trọng số pretrained, không LLM, không API ngoài)
- **Điểm baseline:** chrF = 44.1
- **Điểm BTC (100 điểm):** chrF = 95.0

## Bối cảnh

Một nhóm nghiên cứu ngôn ngữ học đang số hoá kho truyện kể của người Kơ Ru, một cộng đồng nhỏ sống ven
một con sông vùng cao. Tiếng Kơ Ru chỉ còn vài trăm người nói. Nhóm đã ghi chép được khoảng hai vạn câu
kèm bản dịch tiếng Việt do các thầy cô giáo trong bản thực hiện. Phần còn lại của kho truyện chưa có ai
dịch. Nhiệm vụ của bạn là dựng một hệ thống dịch máy Kơ Ru → Việt **chỉ từ dữ liệu song ngữ này**.

Nhóm có ghi lại vài nhận xét ban đầu:

- Tiếng Kơ Ru là ngôn ngữ **chắp dính**: một từ có thể mang nhiều hậu tố nối liền (số nhiều, vai trò
  ngữ pháp, phủ định, thì…). Trật tự câu khác hẳn tiếng Việt, và người kể đôi khi đảo trật tự các thành phần.
- Có những từ **cùng một dạng nhưng nghĩa khác nhau tùy ngữ cảnh**.
- **Tên riêng** (người, vật nuôi…) phải giữ nguyên trong bản dịch.
- Những truyện chưa dịch thường có **câu dài hơn** và **cách kết hợp từ mới** so với phần đã dịch.

(Tiếng Kơ Ru trong bài là ngôn ngữ nhân tạo do BTC tạo ra; bản dịch tiếng Việt là câu tiếng Việt có dấu
chuẩn, viết thường trừ tên riêng, không có dấu câu.)

## Nhiệm vụ

Với mỗi câu tiếng Kơ Ru trong tập test, sinh ra bản dịch tiếng Việt.

## Dữ liệu

| Thư mục | Số câu | `answers.csv`? | Dùng để |
|---|---|---|---|
| `data/train/` | 20 000 | ✅ có | huấn luyện |
| `data/test_public/` | 1 000 | ✅ có (bản sao dev) | chạy pipeline và tự chấm cục bộ |
| tập chấm ẩn | 2 000 | ❌ | chấm điểm chính thức |

Tập chấm ẩn do BTC giữ, **cùng định dạng và cùng phân phối với `test_public`** (gồm cả câu dài hơn train,
tổ hợp từ mới, tên riêng chưa gặp), không có đáp án. Lưu ý: tổ hợp mới và tên riêng mới của tập ẩn **khác** với
của `test_public` — học thuộc `test_public` không giúp được gì. Tại thời điểm chấm, `data/test_public/` được
thay bằng tập ẩn và lời giải của bạn được chạy lại.

```
data/train/data.csv          # id,src            — câu tiếng Kơ Ru
data/train/answers.csv       # id,translation    — bản dịch tiếng Việt
data/test_public/data.csv    # id,src
data/test_public/answers.csv # id,translation,subset   (chỉ có ở bản sao dev; subset ∈ iid | long | compositional)
```

Lấy dữ liệu: tại thư mục bài, chạy `unzip public_data.zip` (tạo ra `data/train/` và `data/test_public/`).

Ví dụ một cặp câu trong train:

```
src:         mobode Zeluno pawu nuzu kawane kara nemuke Nesulo nuguri miyisu
translation: ngày mai Nesulo sẽ yêu anh ấy vì bạn vui vẻ của Zelu đang ngủ ở nhà
```

## Định dạng nộp

File CSV mã hoá UTF-8, đúng hai cột `id,translation`, mỗi `id` của tập test xuất hiện đúng một lần:

```csv
id,translation
pub_00000,bây giờ Petokho đang tìm cô ấy ở công viên
pub_00001,con mèo đen đã ngủ ở nhà
```

Thiếu id, trùng id, id lạ hoặc bản dịch rỗng → bài nộp bị từ chối (không chấm).

## Ràng buộc

- Chỉ dùng dữ liệu trong `data/train/` (và đầu vào `data.csv` của tập test khi suy luận). Không dùng dữ liệu
  ngoài, từ điển ngoài, trọng số pretrained, LLM hay API.
- Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chỉnh tham số — chỉ dùng để tự chấm.
- Được dùng bất kỳ thư viện nào có sẵn (PyTorch, scikit-learn, …); mô hình huấn luyện từ đầu.
- Tổng thời gian huấn luyện + suy luận lúc chấm ≤ 10 phút.

## Chấm điểm

Metric chính là **chrF** (character n-gram F-score, $\beta = 1$) tính trên **toàn corpus**. Khoảng trắng bị bỏ qua.
Với mỗi bậc $n = 1..6$, cộng dồn trên mọi câu số n-gram ký tự khớp giữa bản dịch $h$ và đáp án $r$:

$$P_n = \frac{\sum_i |\,\text{ng}_n(h_i) \cap \text{ng}_n(r_i)\,|}{\sum_i |\text{ng}_n(h_i)|},\qquad
R_n = \frac{\sum_i |\,\text{ng}_n(h_i) \cap \text{ng}_n(r_i)\,|}{\sum_i |\text{ng}_n(r_i)|}$$

$$P = \frac{1}{6}\sum_{n=1}^{6} P_n,\quad R = \frac{1}{6}\sum_{n=1}^{6} R_n,\qquad
\text{chrF} = 100\cdot\frac{(1+\beta^2)\,P\,R}{\beta^2 P + R} = 100\cdot\frac{2PR}{P+R}\quad(\beta = 1).$$

- $\beta = 1$ cân bằng precision và recall: bỏ sót từ và **độn thêm từ** đều bị phạt như nhau (chrF chuẩn của
  sacrebleu dùng $\beta = 2$, vốn thưởng cho việc nối thêm các từ phổ biến vào cuối câu — đề này cố ý không dùng).
- chrF chấm ở mức ký tự nên thưởng một phần cho từ gần đúng (sai dấu, sai một âm tiết), nhưng **sai trật tự**
  làm mất các n-gram dài (n = 4..6) vượt qua ranh giới từ.
- `score.py` còn báo BLEU-4 (mức từ) và chrF riêng cho từng nhóm `iid` / `long` / `compositional` để tham khảo;
  chỉ chrF toàn corpus được dùng để xếp hạng.

Quy đổi ra thang 0–100:

$$\text{Điểm} = 100\cdot\operatorname{clip}\!\left(\frac{\text{chrF} - 44.1}{95.0 - 44.1},\ 0,\ 1\right)$$

Tự chấm cục bộ: `python score.py --pred submission.csv` (mặc định `--split public`).

## Baseline

`baseline.py`: dịch **từng từ** bằng từ điển đồng xuất hiện. Với mỗi từ Kơ Ru nguyên dạng (không tách hậu tố),
chọn các âm tiết tiếng Việt có hệ số Dice cao nhất với nó trên toàn tập train; giữ nguyên trật tự câu nguồn;
từ chưa gặp mà viết hoa thì chép nguyên, còn lại bỏ qua. Đạt chrF ≈ **44.1** (BLEU ≈ 9) trên tập ẩn → 0 điểm.

```
python baseline.py            # đọc data/, ghi submission.csv
python score.py --pred submission.csv
```
