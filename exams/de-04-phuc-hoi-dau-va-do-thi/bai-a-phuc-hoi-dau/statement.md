# Bài A — Phục hồi dấu tiếng Việt

- **Thời gian chạy:** 10 phút (wall-clock) cho toàn bộ huấn luyện + suy luận lúc chấm
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (không word embedding, không mô hình ngôn ngữ, không từ điển/corpus ngoài)
- **Điểm baseline:** độ chính xác âm tiết = 0.728
- **Điểm BTC (100 điểm):** độ chính xác âm tiết = 0.925

## Bối cảnh

Tổng đài chăm sóc khách hàng của một nhà mạng ở Đà Nẵng nhận hàng chục nghìn tin nhắn mỗi ngày. Phần lớn
được gõ **không dấu** trên những chiếc điện thoại cũ: "ma toi bi sot cao", "ban nay con cho khong?",
"mua it dua ve de lay nuoc cot nau che". Hệ thống phân loại tự động của tổng đài hiểu sai liên tục: "má" thành "mã",
"dừa" thành "dưa". Bạn được mời xây dựng bộ **phục hồi dấu** chỉ từ kho câu có dấu nội bộ của công ty.
Cái khó là tin nhắn thật ngày càng đa dạng. Người dùng viết những câu chưa từng gặp, trộn nhiều chủ đề, gõ hoa thường
tùy hứng. Nhiều khi, từ quyết định nghĩa của một âm tiết lại nằm ở cuối câu.

## Nhiệm vụ

Cho một câu tiếng Việt đã bị bỏ toàn bộ dấu (kể cả `đ → d`), hãy khôi phục dạng có dấu của **từng âm tiết**.

- Các âm tiết phân tách bằng dấu cách. Dấu câu có thể dính vào âm tiết (`"ban,`) hoặc đứng riêng (`-`).
- Nhiều âm tiết không dấu có nhiều cách thêm dấu hợp lệ: `ma` → ma / má / mà / mã / mạ; `dua` → dưa / dừa / dứa /
  đua / đũa; `co` → có / cô / cờ / cổ / cỏ / cơ... Có những trường hợp hai âm tiết hai bên **giống hệt nhau** ở mọi
  cách thêm dấu (`mua it dua ve`), và chỉ một cụm từ ở xa hơn trong câu mới quyết định được dạng đúng.

## Dữ liệu

Giải nén tại thư mục bài: `unzip public_data.zip` (tạo thư mục `data/`).

| Thư mục | Số câu | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 40 000 | ✅ (chính là câu có dấu) | huấn luyện |
| `data/test_public/` | 1 500 | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm cục bộ |
| tập chấm ẩn | 3 000 | ẩn | chấm chính thức |

Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` nhưng không có đáp án.

```
data/train/data.csv            # id,text — câu có dấu, sạch, viết hoa chuẩn
data/test_public/data.csv      # id,text — câu KHÔNG dấu, có nhiễu
data/test_public/answers.csv   # id,text — đáp án (chỉ có ở bản sao dev)
```

Đặc điểm cần lưu ý:

- **Train sạch, test nhiễu.** Test có câu viết thường toàn bộ, VIẾT HOA toàn bộ, hoa thường lẫn lộn; dấu câu
  bị bỏ, thêm `...`, `!!`, `:)`, dấu ngoặc kép, gạch nối đứng riêng.
- **Distribution shift có chủ đích.** Tập test chứa **khung câu và mẫu câu không xuất hiện trong train**, câu
  **ghép hai chủ đề khác nhau** (train chỉ ghép trong cùng chủ đề) và **từ hiếm** (từ láy) không có trong train.
  Tập chấm ẩn có cùng *kiểu* shift nhưng dùng **khung câu, mẫu câu và từ hiếm khác** với `test_public`. Học thuộc
  `test_public` không giúp gì cho tập ẩn.
- Corpus gồm 19 nhóm chủ đề (thời tiết, giáo dục, y tế, giao thông, nông nghiệp, kinh tế, gia đình, công nghệ, thể thao,
  ẩm thực, du lịch, môi trường, pháp luật, thiên nhiên, lịch sử – văn học, công sở, xây dựng, hội thoại, hỏi đáp)
  cùng nhiều câu chuyện mua bán đồ dùng, thực phẩm hằng ngày.

## Định dạng nộp

File CSV `submission.csv` (UTF-8) gồm cột `id,text`, mỗi id của tập test đúng một lần:

```csv
id,text
pub_00000,"""Má ơi, dì để - cái bàn ở đâu rồi?"""
pub_00001,không biết bạt loại nào tốt để che mưa cho xe hàng hả ba
```

Phải giữ nguyên số âm tiết và thứ tự của câu đầu vào. Không cần giữ hoa thường hay dấu câu.

## Ràng buộc

- Chỉ dùng dữ liệu trong `data/train` (và đầu vào `data.csv` của tập test để suy luận). Không dùng từ điển,
  corpus hay danh sách từ tiếng Việt bên ngoài.
- **Không** dùng `test_public/answers.csv` để huấn luyện hay tinh chỉnh tham số theo từng câu. File này chỉ dùng để tự chấm.
- Lúc chấm, `data/test_public/` được thay bằng tập ẩn cùng định dạng (không có `answers.csv`). Code phải tự đọc
  `data.csv` và ghi `submission.csv` trong giới hạn 10 phút trên CPU.

## Chấm điểm

Mỗi câu $i$ được chuẩn hoá như sau: NFC, chữ thường, bỏ ký tự không phải chữ/số ở hai đầu mỗi token, bỏ token không chứa
chữ cái. Gọi $g_{i,1..n_i}$ là các âm tiết đáp án và $p_{i,1..m_i}$ là các âm tiết dự đoán:

$$
s_i = \begin{cases} \dfrac{1}{n_i}\sum_{j=1}^{n_i} \mathbb{1}[p_{i,j} = g_{i,j}] & m_i = n_i \\[2mm] 0 & m_i \ne n_i \end{cases}
\qquad
\text{Score} = \frac{1}{N}\sum_{i=1}^{N} s_i
$$

- Trung bình theo **câu** (mỗi câu có trọng số như nhau), nên câu ngắn sai một âm tiết bị phạt nặng hơn câu dài.
- **Không phân biệt hoa thường**, không chấm dấu câu. Nếu số âm tiết dự đoán khác đáp án (thêm, bớt, tách hay gộp từ),
  cả câu được 0.
- Báo thêm (không tính điểm): độ chính xác gộp `syllable_acc_micro`, tỷ lệ câu đúng hoàn toàn `sentence_acc` và
  nửa độ rộng khoảng tin cậy 95% `ci95`.

Quy ra điểm 0–100:

$$\text{Điểm} = 100 \cdot \text{clip}\left(\frac{\text{Score} - 0.728}{0.925 - 0.728},\ 0,\ 1\right)$$

Lệnh tự chấm: `python score.py --pred submission.csv`. Bài nộp sai định dạng (thiếu, trùng hoặc lạ id, text rỗng)
bị từ chối với thông báo lỗi.

## Baseline

`baseline.py` làm như sau: với mỗi âm tiết không dấu (chữ thường), chọn dạng có dấu **xuất hiện nhiều nhất** trong train
(unigram); âm tiết chưa gặp giữ nguyên; chép lại kiểu hoa thường và dấu câu của đầu vào.
Baseline đạt **≈0.728** (sentence accuracy < 1%), gần như câu nào cũng sai ít nhất một từ.
