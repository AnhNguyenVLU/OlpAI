# Bài A — Tìm đúng bức ảnh

- **Thời gian chạy:** 15 phút (wall-clock cho toàn bộ huấn luyện + suy luận lúc chấm)
- **Môi trường:** 4 CPU, 16 GB RAM, không GPU, không internet
- **Mô hình pretrained:** không được dùng (mọi tham số phải học từ dữ liệu BTC cấp)
- **Điểm baseline:** MRR = 0.237
- **Điểm BTC (100 điểm):** MRR = 0.99

## Bối cảnh

Một nhà xuất bản giáo dục đang số hoá kho hình minh hoạ cho sách Toán lớp 1: mỗi hình là vài hình học tô màu
(hình tròn, hình vuông, tam giác, ngôi sao, hình thoi, chữ thập) đặt trên nền giấy có vân. Biên tập viên muốn
gõ một câu mô tả tiếng Việt — *"Hình tròn đỏ ở bên trái hình vuông xanh lam, và không có ngôi sao nào"* — rồi
để máy tìm đúng hình trong kho. Bộ hình mẫu kèm chú thích được chụp trong phòng chụp chuẩn; còn kho cần tìm
lại được chụp lại từ bản in dưới đủ loại đèn: ám vàng, ám xanh, chỗ sáng chỗ tối.

Biên tập viên lâu năm dặn: *"Máy cũ chỉ so màu nên lần nào cũng trả về hình 'na ná'. Đọc cho kỹ từng chữ:
'bên phải' của hình này là 'bên trái' của hình kia, 'không có' khác hẳn 'có', 'hai' khác 'ba'. Và đừng tin
vào màu tuyệt đối dưới ánh đèn lạ."*

## Nhiệm vụ

Mỗi **truy vấn** gồm một chú thích tiếng Việt và **20 ảnh ứng viên**. Đúng **một** ảnh khớp chú thích
hoàn toàn; 19 ảnh còn lại là ảnh "gần đúng": mỗi ảnh **sai ít nhất một chi tiết** so với chú thích (một màu, một dạng
hình, cỡ, số lượng, hướng của một quan hệ vị trí, vai trò của các vật trong một quan hệ, có/không có một vật, hoặc hai
vật bị tráo màu cho nhau). Các ứng viên được vẽ độc lập: bố cục, vật phụ, nền và ánh sáng của mỗi ảnh đều khác nhau.

Hãy xếp hạng 20 ứng viên theo mức độ khớp với chú thích, ảnh khớp nhất đứng đầu.

## Dữ liệu

| Thư mục | Nội dung | Có đáp án? | Dùng để |
|---|---|---|---|
| `data/train/` | 3 000 ảnh, mỗi ảnh 3 chú thích **đúng** (9 000 cặp ảnh–chú thích) | ✅ (chỉ có cặp đúng) | huấn luyện |
| `data/test_public/` | 150 truy vấn × 20 ứng viên (3 000 ảnh) | ✅ `answers.csv` (bản sao dev) | tự chấm cục bộ |
| tập chấm ẩn | 600 truy vấn × 20 ứng viên (12 000 ảnh) | ❌ | chấm chính thức — cùng định dạng và kiểu phân phối với `test_public` |

```
data/train/images/tr_000000.png ...        # ảnh RGB 64x64
data/train/captions.csv                    # image_id,caption  (3 dòng mỗi ảnh)
data/test_public/images/pub_img_000000.png ...
data/test_public/queries.csv               # query_id,caption,candidates (20 image_id cách nhau bởi dấu cách)
data/test_public/answers.csv               # query_id,image_id — bản sao dev, KHÔNG có trong tập chấm ẩn
data/test_public/sample_submission.csv
```

Giải nén dữ liệu tại thư mục bài: `unzip public_data.zip`.
Tập chấm ẩn do BTC giữ, cùng định dạng `test_public` (id dạng `prv_000000`, ảnh `prv_img_000000.png`) nhưng
không có `answers.csv`; lúc chấm nó thay thế `data/test_public/`.

**Ảnh.** Mỗi ảnh có 2–6 vật, không chồng lên nhau. Mỗi vật có **dạng** (hình tròn, hình vuông, tam giác, ngôi sao,
hình thoi, chữ thập), **màu** (đỏ, cam, vàng, xanh lá, xanh dương, tím, hồng, nâu) và **cỡ** (lớn hoặc nhỏ);
vị trí và độ xoay nhỏ ngẫu nhiên. Nền sáng có gradient, đôi khi có vân sọc/ô/chấm và vài đốm bẩn nhỏ.

**Chú thích.** Một hoặc hai mệnh đề (nối bằng "và", "đồng thời", "ngoài ra", "nhưng", hoặc tách câu), mỗi mệnh đề
thuộc một trong các loại sau, với nhiều cách diễn đạt và từ đồng nghĩa (ví dụ "xanh dương"/"xanh lam", "lớn"/"to"):

| Loại | Ví dụ | Nghĩa |
|---|---|---|
| tồn tại | "Có một hình thoi vàng cỡ lớn." | ảnh có **đúng một** vật như mô tả |
| đếm | "Có đúng hai hình tròn đỏ.", "Số vật màu tím trong ảnh là ba." | số vật khớp mô tả bằng **chính xác** con số (từ một đến bốn) |
| phủ định | "Không có ngôi sao nào.", "Không có hình nào màu nâu." | không vật nào khớp mô tả |
| vị trí | "Hình tròn đỏ ở bên trái hình vuông xanh lam.", "Phía trên tam giác cam có một chữ thập hồng.", "Hình thoi tím nằm xa ngôi sao nâu." | so tâm hai vật: trái/phải/trên/dưới theo hướng lệch **trội**; "cạnh" = hai vật đặt sát nhau; "xa" = hai vật cách xa nhau |
| nằm giữa | "Hình thoi tím nằm giữa hình tròn vàng và ngôi sao nâu.", "Hình tròn đỏ không nằm giữa hình vuông xanh và ngôi sao nâu." | theo chiều ngang, tâm vật thứ nhất nằm giữa (hoặc nằm ngoài khoảng giữa) tâm hai vật kia |

Quy ước: một mô tả có thể chỉ nêu dạng, chỉ nêu màu, hoặc cả dạng + màu (+ cỡ). Mô tả trong mệnh đề tồn tại, vị trí,
nằm giữa chỉ có nghĩa khi ảnh có **đúng một** vật khớp mô tả đó; nếu ảnh có không vật nào, hoặc từ hai vật trở lên khớp
mô tả, thì mệnh đề đó sai (ví dụ "Có một hình tròn đỏ" sai với ảnh có hai hình tròn đỏ). Một ảnh khớp chú thích khi mọi
mệnh đề của chú thích đều đúng. Các quan hệ vị trí trong bộ dữ liệu đều **rõ rệt**:
- "A ở bên trái B": tâm A lệch sang trái so với tâm B ít nhất 8 điểm ảnh và độ lệch ngang vượt độ lệch dọc ít nhất
  6 điểm ảnh ("bên phải", "phía trên", "phía dưới" hiểu tương tự theo trục tương ứng);
- "cạnh" ứng với khe hở rất nhỏ giữa hai vật (vài điểm ảnh, tính giữa hai đường tròn ngoại tiếp), "xa" ứng với khe hở từ
  14 điểm ảnh trở lên;
- "A nằm giữa B và C": theo chiều ngang, tâm A nằm trong khoảng giữa tâm B và tâm C và cách mỗi tâm ít nhất 8 điểm ảnh;
  "A không nằm giữa B và C": tâm A nằm ngoài khoảng đó, cách đầu mút gần nhất ít nhất 8 điểm ảnh;
- ở ảnh không khớp, quan hệ bị vi phạm rõ rệt (không có trường hợp sát ngưỡng).

**Lưu ý phân phối.** Ảnh test được chụp dưới ánh sáng khác (ám màu mạnh, độ sáng thay đổi, sáng không đều) —
`test_public` phản ánh đúng dịch chuyển này. Một số tổ hợp **dạng–màu** (ví dụ một màu nào đó của ngôi sao) và một số
kiểu **ghép hai mệnh đề** xuất hiện trong test nhưng **không bao giờ** có trong train; tập chấm ẩn có thể dùng những
tổ hợp mới khác nữa. Chú thích test thiên về hai mệnh đề nhiều hơn train và ảnh test thường đông vật hơn ảnh train (phần lớn có 5–6 vật). Mọi từ ngữ trong chú thích test đều đã xuất
hiện trong train.

## Định dạng nộp

File CSV `submission.csv`, header `query_id,ranking`, mỗi truy vấn đúng một dòng; `ranking` là **đúng 20**
`image_id` của truy vấn đó (mỗi id đúng một lần), cách nhau bởi một dấu cách, ảnh khớp nhất đứng đầu:

```csv
query_id,ranking
pub_000000,pub_img_000444 pub_img_000544 pub_img_000111 ... pub_img_000043
pub_000001,pub_img_000324 pub_img_000407 pub_img_000571 ... pub_img_000187
```

Thiếu / trùng / thừa `query_id`, ranking không đủ 20 id, có id lặp hoặc id không thuộc ứng viên của truy vấn
→ bài nộp **bị từ chối** (score.py báo lỗi).

## Ràng buộc

- Chỉ dùng dữ liệu BTC cấp. Không dữ liệu ngoài, không mô hình pretrained, không LLM/API.
- **Không dùng đáp án `test_public/answers.csv` để huấn luyện hay chọn tham số** — chỉ để tự chấm. Được dùng ảnh và chú
  thích test (không nhãn) lúc suy luận, ví dụ để chuẩn hoá màu.
- Toàn bộ pipeline huấn luyện + suy luận phải chạy xong trong 15 phút trên môi trường chấm.
- Nộp kèm code, trọng số (nếu có) và báo cáo kỹ thuật ngắn.

## Chấm điểm

Với truy vấn $q$, gọi $\text{rank}_q \in \{1, \dots, 20\}$ là vị trí của ảnh đúng trong ranking bạn nộp.
Điểm thô là **Mean Reciprocal Rank** trên $Q$ truy vấn của tập chấm ($Q = 600$ ở tập ẩn):

$$\mathrm{MRR} = \frac{1}{Q}\sum_{q=1}^{Q} \frac{1}{\text{rank}_q}$$

- Xếp ngẫu nhiên cho MRR ≈ 0.18; ảnh đúng luôn đứng đầu cho MRR = 1. `score.py` báo thêm **R@1**
  (tỉ lệ truy vấn có ảnh đúng đứng đầu) và **nửa độ rộng** khoảng tin cậy 95% của MRR (`mrr_ci95`; khoảng tin cậy là
  MRR ± `mrr_ci95`).
- MRR thưởng nặng việc đưa ảnh đúng lên **hạng 1** (hạng 2 chỉ còn 0.5 điểm thô).

Quy đổi ra thang 0–100:

$$\text{điểm} = 100\cdot\mathrm{clip}\left(\frac{\mathrm{MRR} - 0.237}{0.99 - 0.237},\ 0,\ 1\right)$$

Tự chấm: `python score.py` (mặc định chấm `submission.csv` trên `test_public`).

## Baseline

`baseline.py`: chia mỗi ảnh cho màu trung vị của nó (bù ám màu một cách thô), lấy histogram màu 4×4×4 của các điểm
ảnh lệch xa nền; với mỗi từ trong chú thích train, "mẫu màu" của từ = histogram trung bình của các ảnh có từ đó trừ
histogram trung bình chung. Điểm của một ảnh ứng viên = tổng tích vô hướng giữa histogram của ảnh và mẫu của các từ
trong chú thích (vế câu có chữ "không" thì đổi dấu). Nó chỉ "nhìn" màu nên gần như không phân biệt được các ảnh
gần đúng: điểm **MRR = 0.237** trên tập ẩn (xếp ngẫu nhiên ≈ 0.18). Điểm BTC **MRR = 0.99** đạt được bằng mô hình
cổ điển (numpy/scipy/scikit-learn, CPU, vài phút) trong giới hạn thời gian; tối đa của metric là 1.0.
