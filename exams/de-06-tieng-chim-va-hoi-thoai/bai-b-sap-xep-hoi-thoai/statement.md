# Bài B — Tổng đài bị xáo trộn

- **Thời gian chạy:** 15 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (không word embedding, không PhoBERT/LLM, không API ngoài)
- **Điểm baseline:** độ chính xác thứ tự theo cặp = 0.833
- **Điểm BTC (100 điểm):** độ chính xác thứ tự theo cặp = 0.926

## Bối cảnh

Trung tâm chăm sóc khách hàng của một tập đoàn dịch vụ phục vụ khách qua kênh chat ở mười lĩnh vực: ngân hàng, di động,
internet, truyền hình, điện lực, giao hàng, thương mại điện tử, ví điện tử, hàng không và phòng khám. Mỗi cuộc hội thoại là
một chuỗi **lượt** (tin nhắn) giữa khách hàng và tổng đài viên. Trong đợt chuyển hệ thống, các lượt của mọi cuộc hội thoại
bị **xáo trộn**: không còn dấu thời gian, không còn nhãn người nói; chỉ còn biết **hai lượt đầu tiên** thật sự của mỗi cuộc.

Một giám sát viên lâu năm xem qua dữ liệu rồi dặn: *"Tổng đài viên nói theo kịch bản, lịch sự, kết thúc bằng 'ạ'; khách thì
gõ tắt, đôi khi nhắn hai tin liền. Đừng đọc từng tin riêng lẻ: hỏi thì phải có đáp, 'mã đó', 'như em vừa nói' phải trỏ về
một tin trước, và việc nào cũng có trình tự: xác minh rồi mới làm, làm xong mới chốt. Có cuộc khách nêu hai việc một lúc, bên
tổng đài giải quyết lần lượt. Còn bộ dữ liệu mới thì khách gõ bằng điện thoại, nhiều nghiệp vụ và cách nói mà bộ cũ chưa có."*

## Nhiệm vụ

Với mỗi cuộc hội thoại có $n$ lượt (8 ≤ $n$ ≤ 18), khôi phục **thứ tự thời gian** của các lượt. Với lượt thứ $i$ trong file
(thứ tự xáo trộn, $i = 0, \dots, n-1$), dự đoán $P[i]$ = vị trí thật của nó trong cuộc hội thoại (0 = lượt đầu tiên).
$P$ phải là một **hoán vị** của $\{0, 1, \dots, n-1\}$; với hai lượt đầu đã cho, $P$ đúng phải có giá trị 0 và 1 tại hai chỉ số đó.

## Dữ liệu

| Thư mục | Số hội thoại | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 4 000 | ✅ `answers.csv` | huấn luyện |
| `data/test_public/` | 1 000 | ✅ `answers.csv` (bản sao dev) | chạy pipeline, tự chấm cục bộ |
| tập chấm ẩn | 2 000 | ❌ | chấm chính thức — cùng định dạng và phân phối với `test_public` |

```
data/train/turns.csv            # id,turn,text      mỗi dòng một lượt (thứ tự xáo trộn)
data/train/prefix.csv           # id,first,second   chỉ số `turn` của lượt thứ nhất và lượt thứ hai THẬT
data/train/answers.csv          # id,order          order[i] = vị trí thật của lượt i, các số cách nhau bởi dấu cách
data/test_public/turns.csv
data/test_public/prefix.csv
data/test_public/answers.csv    # bản sao dev — KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.

- `id`: mã hội thoại (`tr_000000` ở train, `pub_000000` ở test_public); `turn`: chỉ số lượt trong thứ tự xáo trộn (0 … $n-1$);
  `text`: nội dung tin nhắn, tiếng Việt, một dòng.
- Ví dụ (trích đúng từ `test_public`): `prefix.csv` có dòng `pub_000007,12,1` nghĩa là lượt `turn = 12` là lượt đầu tiên và lượt `turn = 1`
  là lượt thứ hai của hội thoại `pub_000007`; `answers.csv` có `order = "16 1 8 9 12 ..."` nghĩa là lượt `turn = 0` có vị trí thật là 16,
  lượt `turn = 1` có vị trí thật là 1, lượt `turn = 2` có vị trí thật là 8, ... (vị trí tính từ 0; hội thoại này có $n = 18$ lượt).
- **Không có nhãn người nói.** Mỗi cuộc có hai bên (khách và tổng đài viên) nhưng một bên có thể gửi hai tin liên tiếp.
- Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` nhưng **chỉ có `turns.csv` và `prefix.csv`**; lúc chấm nó thay thế
  `data/test_public/`, nên code phải đọc `turns.csv`/`prefix.csv` thay vì `answers.csv`.

Train và test khác nhau ở những điểm sau (bạn phải tự phát hiện bằng cách so sánh `train` với `test_public`):

- khách ở train phần lớn viết có dấu, đúng chính tả; khách ở test có tỉ lệ lớn tin **không dấu**, **teencode**, lỗi gõ;
- một số **nghiệp vụ** (loại việc khách cần giải quyết) và một số **mẫu câu** của cả hai bên chỉ xuất hiện ở test;
  phần mới của tập chấm ẩn **khác** phần mới của `test_public`, nên điểm trên `test_public` thường cao hơn điểm thật; không được dùng
  đáp án `test_public` để huấn luyện (xem Ràng buộc).

## Định dạng nộp

File `submission.csv` gồm các cột `id,order`, mỗi hội thoại đúng một dòng; `order` là $n$ số nguyên cách nhau bởi dấu cách:

```csv
id,order
pub_000000,2 3 1 4 5 6 7 8 9 0 10 11 12 13 14
pub_000001,2 3 4 5 6 7 8 9 10 11 12 0 13 14 1 15 16 17
```

(Hai dòng đầu của `sample_submission.csv`: hội thoại `pub_000000` có $n = 15$ lượt nên `order` có 15 số, `pub_000001` có $n = 18$ lượt nên có 18 số.)

- Mỗi `id` của `test_public/turns.csv` xuất hiện đúng một lần. Thiếu id, trùng id, id lạ, thiếu cột hoặc `order` chứa ký tự
  không phải số nguyên → bài nộp **bị từ chối** (score.py báo lỗi, exit code 1).
- Nếu `order` là các số nguyên nhưng **không phải hoán vị hợp lệ** của $0, \dots, n-1$ (sai độ dài, trùng, thiếu, số ngoài khoảng,
  đánh số từ 1) thì hội thoại đó được **0 điểm** (các hội thoại khác vẫn được tính).
- `sample_submission.csv` (giữ hai lượt đầu, các lượt còn lại theo thứ tự xáo trộn) là ví dụ về một bài nộp hợp lệ.

## Ràng buộc

- Chỉ dùng dữ liệu trong `train/`. Không dữ liệu ngoài, không pretrained, không LLM/API. Được tự viết luật chuẩn hoá văn bản.
- **Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn mô hình** (kể cả làm dữ liệu bổ sung, hiệu chỉnh hay stacking,
  **kể cả sao chép hoặc nhúng dữ liệu/trọng số lấy từ `test_public` vào mã nộp**) — chỉ để tự chấm.
  Được dùng văn bản test (không nhãn) lúc suy luận, ví dụ để xây từ vựng hoặc thích nghi miền không giám sát.
- **Khi chấm, BTC chạy lại toàn bộ pipeline (huấn luyện + suy luận) từ mã nộp trong một thư mục chỉ có `data/train/` và tập chấm ẩn**
  (đặt ở `data/test_public/`, chỉ có `turns.csv` và `prefix.csv`; không có `answers.csv`, `sample_submission.csv`). Vì vậy mã phải tự huấn luyện từ
  `train/`, không kèm trọng số hay dữ liệu tính sẵn. Mã chạy được bằng một lệnh duy nhất (ví dụ `python solution.py`; tên file do BTC quy định)
  và ghi `submission.csv` vào thư mục hiện tại; file dự đoán do BTC tạo khi chạy lại nên không cần nộp kèm. BTC rà soát mã của các đội đứng đầu.
- Toàn bộ pipeline huấn luyện + suy luận chạy xong trong 15 phút. Nộp kèm code và báo cáo kỹ thuật ngắn.

## Chấm điểm

Với hội thoại $d$ có $n_d$ lượt, gọi $M_d = n_d(n_d-1)/2$ là số cặp lượt và $I_d$ là số cặp được xếp **ngược** so với thứ tự thật
(cặp $(i,j)$ mà thứ tự tương đối của $P[i], P[j]$ khác thứ tự tương đối của vị trí thật). Điểm thô là **độ chính xác thứ tự theo cặp**
trung bình theo hội thoại ($N$ = số hội thoại; hoán vị không hợp lệ cho $s_d = 0$):

$$s_d = 1 - \frac{I_d}{M_d}, \qquad \text{score} = \frac{1}{N}\sum_{d=1}^{N} s_d$$

- Mỗi cặp nặng như nhau; thứ tự hoàn hảo cho $s_d = 1$, thứ tự đảo ngược hoàn toàn cho $0$, thứ tự ngẫu nhiên cho $\approx 0.5$.
  Vì hai lượt đầu đã biết, các cặp liên quan đến chúng luôn đúng nếu bạn giữ nguyên chúng (với $n = 16$ là 29 trên 120 cặp);
  giữ hai lượt đầu rồi để nguyên thứ tự xáo trộn cho các lượt còn lại chỉ được ≈ 0.63.
- Một lượt đặt sai chỗ làm sai nhiều cặp cùng lúc: lỗi ở "lượt chốt" hay "lượt xác minh" rất đắt.
- `score.py` báo thêm tỉ lệ hội thoại **xếp đúng hoàn toàn** (`exact_match`) và số hội thoại không hợp lệ để tham khảo.

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\text{score} - 0.833}{0.926 - 0.833},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

Khoảng cách từ baseline đến điểm BTC chỉ 0.093 điểm thô, nên 0.01 điểm thô tương đương ≈ 11 điểm quy đổi. Điểm tự chấm trên `test_public`
**có thể lệch so với tập ẩn cỡ ±0.02 điểm thô; với lời giải gần baseline thường cao hơn khoảng 10 điểm** (baseline: ≈ 0.843 trên `test_public`
so với 0.833 trên tập ẩn). Đừng chỉ dựa vào `test_public` để đánh giá hay chọn mô hình.

## Baseline

`baseline.py`: TF-IDF theo **từ đơn** (500 từ phổ biến nhất, văn bản gốc) rồi Ridge hồi quy **vị trí tương đối** $P[i]/(n-1)$ của từng lượt;
mỗi hội thoại giữ hai lượt đầu đã cho, các lượt còn lại sắp theo vị trí dự đoán tăng dần. Baseline nhìn từng tin riêng lẻ nên chỉ nhận ra
"giai đoạn" của cuộc hội thoại (chào, hỏi mã, chờ, chốt, chào tạm biệt) và hụt khi gặp văn phong lạ ở test. Điểm: score = **0.833** trên tập ẩn
(≈ 0.843 trên `test_public`). Điểm BTC **0.926** đạt được bằng mô hình cổ điển (numpy/scikit-learn, CPU) trong giới hạn thời gian;
tối đa của metric là 1.0.
