# Bài B — Tuần sau học gì?

- **Thời gian chạy:** 15 phút (wall-clock) cho toàn bộ huấn luyện + suy luận lúc chấm
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng
- **Điểm baseline:** NDCG@10 = 0.034
- **Điểm BTC (100 điểm):** NDCG@10 = 0.282

## Bối cảnh

"HọcMọiNơi" là một nền tảng học trực tuyến của một startup ở Đà Nẵng với hàng chục nghìn người học và
1 200 khoá học — từ lập trình Python, tiếng Anh đến tài chính cá nhân. Trang chủ hiện chỉ hiển thị "Top
khoá hot", nên khoá hay của giảng viên mới chìm nghỉm còn người học thì bỏ đi vì gợi ý chẳng liên quan.
Ban giám đốc giao cho đội bạn xây bộ xếp hạng mới: với mỗi người học, chọn **10 khoá** họ có khả năng
**ghi danh trong tuần tới** nhất.

## Nhiệm vụ

Dữ liệu là nhật ký tương tác 52 tuần (tuần 0..51). Với mỗi người dùng trong `test_users.csv`, xếp hạng
**top-10 khoá học** mà người đó sẽ ghi danh trong **tuần 52** (ngày 364..370). Không tính các khoá người
dùng đã ghi danh trước đó (ghi danh lại không bao giờ xảy ra). Danh sách test chỉ gồm người có ít nhất
một lượt ghi danh trong tuần 52.

Một vài đặc điểm của nền tảng:

- Khoá học thuộc các **lộ trình** (track): học xong khoá cấp thấp mới hay học tiếp khoá cấp cao hơn.
  Catalog chỉ công bố **một phần** quan hệ tiên quyết (`prerequisite_id`), phần còn lại phải tự suy ra.
- Có **popularity bias**: khoá đang hot được hiển thị nhiều hơn nên càng hot.
- Khoá mới ra mắt được đẩy lên trang chủ; **một số khoá ra mắt đúng tuần 52** nên không có lịch sử tương
  tác nào (cold-start) nhưng vẫn có trong catalog.
- Người dùng cần gợi ý là **những người khác** với người dùng trong `train/` (nhưng cùng hành vi); lịch
  sử tuần 0..51 của họ nằm trong thư mục test.

## Dữ liệu

| Thư mục | Nội dung | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | `users.csv` (6 000 người, id `U*****`), `courses.csv`, `interactions.csv` (tuần 0..51) | — | huấn luyện, tự tạo nhãn theo thời gian |
| `data/test_public/` | 4 000 người khác (id `pub_*****`): `users.csv`, `interactions.csv` (tuần 0..51), `test_users.csv` (~650 người) | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm |
| tập chấm ẩn | cùng cấu trúc, 4 000 người khác nữa (id `prv_*****`) | — | **chấm chính thức** |

```
data/train/users.csv              user_id, signup_week, occupation, age_group, region
data/train/courses.csv            course_id, topic, level (1-3), instructor_id, duration_hours, is_free,
                                  release_week (≤0: có từ trước; 52: ra mắt tuần mục tiêu), prerequisite_id (có thể rỗng)
data/train/interactions.csv       user_id, course_id, day, event ∈ {view, enroll, complete, rate}, rating (1-5, chỉ với rate)
data/test_public/users.csv        như train/users.csv
data/test_public/interactions.csv như train/interactions.csv (lịch sử của nhóm test)
data/test_public/test_users.csv   user_id — người cần gợi ý
data/test_public/answers.csv      user_id, course_id — bản sao dev, KHÔNG có trong tập chấm ẩn
```

Tuần của một sự kiện là `day // 7`. **Lấy dữ liệu:** tại thư mục bài chạy `unzip public_data.zip`.
Tập chấm ẩn do BTC giữ, cùng định dạng `test_public`, không có đáp án. Lúc chấm, `data/test_public/`
được thay bằng tập ẩn — code của bạn phải đọc lịch sử người dùng từ thư mục test.

## Định dạng nộp

File `submission.csv`, header `user_id,course_ids`; `course_ids` là **đúng 10** mã khoá **khác nhau**,
cách nhau bởi một khoảng trắng, theo thứ tự ưu tiên giảm dần:

```csv
user_id,course_ids
pub_00012,C0593 C0320 C1101 C0044 C0980 C0007 C0815 C0140 C0661 C0302
pub_00031,C1061 C0011 C0593 C0412 C0870 C0320 C0019 C0904 C0150 C0703
```

Thiếu / trùng / thừa `user_id`, ô rỗng, không đủ 10 mã, mã trùng hoặc không có trong `courses.csv`
→ bài nộp bị từ chối.

## Ràng buộc

- Chỉ dùng dữ liệu BTC cấp. **Không** dùng đáp án `test_public/answers.csv` để huấn luyện — nó chỉ để tự chấm.
- Không pretrained weights / LLM / API ngoài. Huấn luyện + suy luận ≤ 15 phút trên 4 CPU.

## Chấm điểm

Với người dùng $u$, gọi $R_u$ là tập khoá thực sự ghi danh trong tuần 52, $c_{u,1..10}$ là danh sách nộp:

$$\text{DCG@10}(u) = \sum_{i=1}^{10} \frac{\mathbb{1}[c_{u,i} \in R_u]}{\log_2(i+1)},\qquad
\text{NDCG@10}(u) = \frac{\text{DCG@10}(u)}{\sum_{i=1}^{\min(|R_u|,10)} \frac{1}{\log_2(i+1)}} .$$

$$\text{Điểm thô} = \frac{1}{|U_{test}|}\sum_{u} \text{NDCG@10}(u) \in [0,1].$$

- Đúng khoá ở vị trí 1 có giá trị gấp ~3.3 lần ở vị trí 10; người chỉ ghi danh 1 khoá mà khoá đó đứng
  đầu danh sách được 1.0.
- `score.py` báo thêm **Recall@10** $= \frac{1}{|U|}\sum_u |R_u \cap \{c_{u,i}\}| / |R_u|$ (chỉ để tham khảo).
- Quy ra điểm 0–100: `Threshold(baseline=0.034, target=0.282)`,

$$\text{Điểm} = 100\cdot\text{clip}\left(\frac{\text{Điểm thô}-0.034}{0.282-0.034},\,0,\,1\right).$$

Tự chấm: `python score.py --pred submission.csv` (mặc định `--split public`).

## Baseline

`baseline.py` gợi ý cho mọi người cùng một danh sách: 10 khoá có **tổng lượt ghi danh nhiều nhất** trong
`train/interactions.csv` (bỏ các khoá người đó đã ghi danh). Điểm thô: **0.034** NDCG@10 trên tập ẩn
(≈ 0.043 trên `test_public`).

Điểm BTC **0.282** đến từ một mô hình học có giám sát chạy trên CPU trong vòng một phút, dùng nhãn tự tạo
theo thời gian. Không phải cận trên — tối đa là 1.0.
