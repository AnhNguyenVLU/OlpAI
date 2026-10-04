# Đề 04 — Phục hồi dấu tiếng Việt & Dự đoán liên kết trên đồ thị

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, khó hơn
đề tham khảo một chút. Đề có distribution shift có chủ đích. Ở bài A là khung câu mới, mơ hồ phụ thuộc ngữ cảnh xa,
câu ghép liên chủ đề và từ hiếm. Ở bài B là tỷ lệ hợp tác liên ngành tăng và tác giả cold-start; bài B còn có bẫy
rò rỉ thời gian nếu chia validation sai.

| Bài | Đề chính thức | Lĩnh vực | Metric chính | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Phục hồi dấu | [bai-a-phuc-hoi-dau/statement.md](bai-a-phuc-hoi-dau/statement.md) | NLP (gán nhãn chuỗi) | độ chính xác âm tiết, trung bình theo câu (↑) | 0.728 → 0.925 | 10 phút |
| B — Gợi ý hợp tác nghiên cứu | [bai-b-du-doan-lien-ket/statement.md](bai-b-du-doan-lien-ket/statement.md) | ML trên đồ thị (link prediction, xếp hạng) | MRR trên 100 ứng viên (↑), kèm Hits@10 | 0.214 → 0.330 | 15 phút |

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài là $100\cdot\operatorname{clip}\big((\text{điểm thô} - \text{baseline}) / (\text{Điểm BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn**. Tập ẩn do BTC giữ, cùng định dạng và cùng kiểu phân phối với `test_public`, không có đáp án.
- Chỉ dùng dữ liệu BTC cấp. **Không** dùng dữ liệu ngoài (kể cả từ điển tiếng Việt), **không** dùng trọng số pretrained,
  LLM hay API ngoài, **không** dùng `test_public/answers.csv` để huấn luyện (file này chỉ để tự chấm).
- Mỗi bài nộp: code huấn luyện và suy luận (chạy lại được trong thời gian quy định, trên CPU), trọng số mô hình,
  file dự đoán `submission.csv`, báo cáo kỹ thuật ngắn (≤ 2 trang: tiền xử lý, mô hình, cách validation, kết quả).
- Gợi ý phân bổ thời gian: 30 phút đọc đề và chạy baseline cả hai bài, khoảng 2.5 giờ cho mỗi bài, 30 phút cuối để
  chạy lại pipeline từ đầu, kiểm tra định dạng và viết báo cáo. Ở bài A, mô hình n-gram cho kết quả khá nhanh nhưng sẽ
  sớm chạm trần. Ở bài B, hãy dựng đúng quy trình validation theo thời gian trước khi làm feature engineering.

## Cách chạy

```bash
cd bai-a-phuc-hoi-dau            # hoặc bai-b-du-doan-lien-ket
unzip public_data.zip            # tạo data/train và data/test_public
python baseline.py               # ghi submission.csv (đọc data/test_public)
python score.py --pred submission.csv   # tự chấm trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}` và
`score.py --data DIR --pred FILE --split {public,private}` (`private` chỉ dùng được ở máy BTC).
