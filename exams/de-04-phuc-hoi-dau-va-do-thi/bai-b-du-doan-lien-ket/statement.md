# Bài B — Gợi ý hợp tác nghiên cứu

- **Thời gian chạy:** 15 phút (wall-clock) cho toàn bộ huấn luyện + suy luận lúc chấm
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** MRR = 0.214
- **Điểm BTC (100 điểm):** MRR = 0.330

## Bối cảnh

Quỹ Phát triển Khoa học Quốc gia muốn xây dựng tính năng "Gợi ý cộng sự" cho cổng thông tin nhà khoa học.
Mỗi nhà nghiên cứu khi đăng nhập sẽ thấy danh sách những người mà họ **có khả năng sẽ hợp tác lần đầu** trong
thời gian tới. Quỹ trao cho bạn toàn bộ lịch sử công bố đồng tác giả trong nước giai đoạn 2000–2023 của
8 000 nhà nghiên cứu thuộc 8 lĩnh vực công nghệ thông tin. Ban giám khảo giữ lại dữ liệu 2024–2025 để kiểm tra
xem mô hình của bạn có xếp những người *thật sự* đã bắt tay viết bài chung lên đầu hay không.

## Nhiệm vụ

Mỗi **truy vấn** là một tác giả $u$ kèm đúng **100 ứng viên**. Trong các ứng viên có từ 1 đến 3 người là
**cộng sự mới** của $u$, tức lần đầu cùng đứng tên một bài báo trong 2024–2025 và chưa từng hợp tác đến hết 2023.
Những người còn lại không hợp tác với $u$ trong 2024–2025. Hãy cho mỗi cặp (truy vấn, ứng viên) một điểm
số; điểm càng cao nghĩa là càng có khả năng hợp tác.

Âm tính được chọn **khó**: khoảng 35% là láng giềng bậc 2 của $u$ (cộng sự của cộng sự), 25% cùng nhóm chủ đề thực tế,
12% cùng trường, 8% tác giả mới vào nghề 2024–2025, phần còn lại lấy ngẫu nhiên theo bậc (để phân phối bậc của
âm tính gần với dương tính). Không ứng viên nào là cộng sự cũ của $u$. Một số ứng viên cũng là tác giả truy vấn khác
trong cùng tập; tỷ lệ này được BTC cân bằng giữa dương và âm nên không mang thông tin.

## Dữ liệu

Giải nén tại thư mục bài: `unzip public_data.zip` (tạo thư mục `data/`).

| Thư mục | Kích thước | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 8 000 tác giả, ~33 nghìn bài báo, ~82 nghìn cặp đồng tác giả (2000–2023) | — | huấn luyện |
| `data/test_public/` | 1 000 truy vấn × 100 ứng viên | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm |
| tập chấm ẩn | 2 000 truy vấn × 100 ứng viên | ẩn | chấm chính thức |

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` nhưng không có `answers.csv`.

```
data/train/nodes.csv          author_id,field,institution,subfield,start_year
data/train/papers.csv         paper_id,year,authors          # authors: "A00012;A03311;A04120"
data/train/edges.csv          u,v,first_year,last_year,n_papers   # tổng hợp sẵn từ papers.csv
data/test_public/queries.csv     query_id,author_id
data/test_public/candidates.csv  query_id,candidate_id
data/test_public/answers.csv     query_id,candidate_id          # các cộng sự mới thật sự (chỉ bản sao dev)
```

- `field`: 8 lĩnh vực; `institution`: 48 trường/viện; `subfield`: chủ đề con tự khai báo, **có nhiễu** (khoảng
  1/4 tác giả khai không khớp nhóm thực tế mình làm việc); `start_year`: năm bắt đầu sự nghiệp nghiên cứu.
  Tác giả bắt đầu năm 2024–2025 **không có cạnh nào** trong train (cold-start).
- `test_public` và tập chấm ẩn được lấy từ hai **giai đoạn 2024–2025 khác nhau** của cùng một cộng đồng, tức là
  cùng lịch sử đến 2023 và cùng quy luật hình thành hợp tác. Điểm trên `test_public` vì thế là ước lượng tốt cho
  tập ẩn, nhưng các cặp trong `answers.csv` của `test_public` không phải đáp án của tập ẩn.

**Những điều cần biết** (cũng là gợi ý):

- Dữ liệu có **thời gian**. Để tạo tập huấn luyện có nhãn, hãy tự cắt mốc trong train
  (ví dụ đồ thị đến 2021 thì nhãn là các cặp mới trong 2022–2023) và mô phỏng lại cách lấy ứng viên như trên.
  Chia cạnh ngẫu nhiên (không theo thời gian) sẽ rò rỉ tương lai và cho validation ảo.
- Giai đoạn 2024–2025 có **shift**: tỷ lệ hợp tác liên ngành tăng lên đáng kể so với trước. Ngoài ra có truy vấn và
  ứng viên cold-start (~9% truy vấn không có cạnh nào trong train).
- Hợp tác không chỉ đến từ "bạn của bạn": giữa một số trường, một số nhóm chủ đề có quan hệ hợp tác lâu dài.

## Định dạng nộp

`submission.csv` gồm cột `query_id,candidate_id,score`. File phải có **đúng** mọi cặp trong `candidates.csv`, mỗi cặp
một lần; `score` là số thực hữu hạn:

```csv
query_id,candidate_id,score
pub_00000,A01234,0.8731
pub_00000,A00077,0.0412
```

## Ràng buộc

- Chỉ dùng dữ liệu trong `data/train` và danh sách truy vấn/ứng viên của tập đang dự đoán.
- **Không** dùng `test_public/answers.csv` để huấn luyện, làm cạnh bổ sung cho đồ thị hay chọn tham số theo từng truy vấn.
  File này chỉ dùng để tự chấm.
- Không khai thác cấu trúc file (thứ tự dòng, tần suất ứng viên xuất hiện trong nhiều truy vấn...). Thứ tự
  ứng viên đã được xáo trộn.
- Lúc chấm, `data/test_public/` được thay bằng tập ẩn cùng định dạng (không có `answers.csv`). Code phải chạy xong
  trong 15 phút trên CPU.

## Chấm điểm

Với truy vấn $q$, tập dương $P_q$, tập âm $N_q$ và điểm $s(\cdot)$, hạng của một dương $p$ được tính như sau (các dương
khác không tính, hoà điểm được tính một nửa):

$$
\text{rank}_q(p) = 1 + \sum_{n \in N_q} \mathbb{1}[s(n) > s(p)] + \tfrac{1}{2}\sum_{n \in N_q} \mathbb{1}[s(n) = s(p)]
$$

$$
\text{RR}_q = \frac{1}{\min_{p \in P_q} \text{rank}_q(p)}, \qquad
\text{MRR} = \frac{1}{|Q|}\sum_{q} \text{RR}_q, \qquad
\text{Hits@10} = \frac{1}{|Q|}\sum_q \mathbb{1}\Big[\min_{p} \text{rank}_q(p) \le 10\Big]
$$

- MRR là metric chính. Đưa *một* cộng sự thật lên hạng 1 được 1.0, hạng 2 được 0.5, hạng 10 chỉ còn 0.1. Vì vậy metric
  thưởng rất mạnh cho việc xếp chính xác ở đầu danh sách.
- Hoà điểm bị tính hạng trung bình: cho mọi ứng viên cùng một điểm thì hạng ≈ 50, RR ≈ 0.02. Đây là điểm yếu
  của Common Neighbors, vì rất nhiều cặp có CN = 0 hoặc bằng nhau.
- Báo thêm (không tính điểm): Hits@10 và `mrr_ci95`, tức nửa độ rộng khoảng tin cậy 95% của MRR.

Quy ra điểm 0–100:

$$\text{Điểm} = 100 \cdot \text{clip}\left(\frac{\text{MRR} - 0.214}{0.330 - 0.214},\ 0,\ 1\right)$$

Lệnh tự chấm: `python score.py --pred submission.csv`. Bài nộp thiếu, thừa hoặc trùng cặp, hay có score rỗng/NaN/inf
đều bị từ chối.

## Baseline

`baseline.py` chấm điểm bằng **số cộng sự chung** (Common Neighbors) của $u$ và ứng viên trên đồ thị đồng tác giả đến 2023.
Cách này đạt MRR ≈ 0.214 (Hits@10 ≈ 0.45). Âm tính bậc 2 cũng có cộng sự chung, nên CN đơn thuần không đủ.
