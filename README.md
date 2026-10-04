# OlpAI — Bộ đề luyện thi Olympic Trí tuệ nhân tạo Sinh viên

Bộ 5 đề luyện thi cho **Olympic Trí tuệ nhân tạo Sinh viên Việt Nam (OlpAI)**. Nội dung bám theo đề tham khảo
OlpAI'25: dịch máy, nhận dạng ngôn ngữ ký hiệu, ba mảng ML / CV / NLP. Cách trình bày theo khung đề
[IOAI 2026 — Individual Contest](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest).

Độ khó được đẩy **nhỉnh hơn đề tham khảo**. Mỗi bài cài sẵn ít nhất một "bẫy" thực tế: distribution shift,
nhiễu nhãn, rò rỉ đặc trưng, cold-start, dữ liệu ít. Chỉ đội nào khám phá dữ liệu kỹ và validation đúng cách
mới đạt điểm cao.

## Danh sách đề

| Đề | Bài A | Bài B |
|---|---|---|
| [01 — Dịch máy & Ngôn ngữ ký hiệu](exams/de-01-dich-may-va-ky-hieu) | NLP · Dịch ngôn ngữ thiểu số → tiếng Việt (chrF) | CV · Nhận dạng ký hiệu tay từ keypoints, người ký mới (macro-F1) |
| [02 — Dự báo & Ý định](exams/de-02-du-bao-va-y-dinh) | ML · Dự báo phụ tải điện theo phân vị (pinball loss) | NLP · Ý định đa nhãn, tin nhắn không dấu/teencode (F1) |
| [03 — Phân đoạn & Tín dụng](exams/de-03-phan-doan-va-tin-dung) | CV · Phân đoạn ruộng lúa + đếm thửa, đổi mùa (mIoU) | ML · Dự đoán vỡ nợ vi mô, rò rỉ + drift (ROC-AUC) |
| [04 — Phục hồi dấu & Đồ thị](exams/de-04-phuc-hoi-dau-va-do-thi) | NLP · Phục hồi dấu tiếng Việt (độ chính xác âm tiết) | ML · Gợi ý hợp tác nghiên cứu, link prediction (MRR) |
| [05 — Few-shot & Xếp hạng](exams/de-05-few-shot-va-xep-hang) | CV · Chữ "kiểu Nôm" 20-way 5-shot, lớp mới (accuracy) | ML · Gợi ý khoá học top-10 (NDCG@10) |

Mỗi đề gồm 2 bài, thi theo đội 3 người trong **6 giờ**, mỗi bài 100 điểm.

## Cấu trúc

```
exams/de-0X-<tên>/
  README.md               tổng quan đề, quy chế, cách chạy
  bai-{a,b}-<tên>/
    statement.md          đề chính thức
    public_data.zip       dữ liệu phát cho thí sinh: train/ + test_public/ (kèm đáp án để tự chấm)
    baseline.py           lời giải cơ sở (= 0 điểm)
    score.py              chấm điểm, quy ra thang 0–100
common/scoring.py         tiện ích chấm điểm dùng chung
scripts/smoke_test.sh     chạy baseline + chấm cho mọi bài (dùng trong CI)
```

## Dành cho sinh viên

```bash
pip install -r requirements.txt          # Python ≥ 3.11
cd exams/de-01-dich-may-va-ky-hieu/bai-a-dich-may
unzip public_data.zip                    # tạo data/train/ và data/test_public/
python baseline.py                       # ghi submission.csv
python score.py --pred submission.csv    # tự chấm trên test_public
```

Quy định chung: chỉ dùng dữ liệu được cấp. **Không** dùng dữ liệu ngoài, trọng số pretrained, LLM hay API.
**Không** dùng đáp án `test_public` để huấn luyện (chỉ dùng để tự chấm).
Được dùng thêm PyTorch và các thư viện khác, miễn là chạy trong giới hạn thời gian ghi ở đề.

## Dành cho giáo viên / BTC

Toàn bộ dữ liệu là **tổng hợp**. Tập chấm ẩn (`data/_private/`) **không** nằm trong repo này. Bộ công cụ giáo viên
gồm generator, hướng dẫn sinh tập ẩn bằng seed bí mật, lời giải tham khảo và mốc điểm. Bộ này được phát riêng để
sinh viên không suy ngược ra được đáp án. Khi có tập ẩn, chấm bài của một đội như sau:

```bash
python <lời giải của đội>.py --split private --out team.csv
python score.py --split private --pred team.csv
```

Kiểm tra nhanh toàn bộ repo: `scripts/smoke_test.sh` (khoảng 30 giây).
