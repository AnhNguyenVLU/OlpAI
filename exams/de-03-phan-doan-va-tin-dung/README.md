# Đề 03 — Phân đoạn ruộng lúa & Tín dụng vi mô

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, độ khó nhỉnh hơn
đề tham khảo: dịch chuyển miền theo mùa, mây che, rò rỉ nhãn, dữ liệu thiếu không ngẫu nhiên và drift theo thời gian.

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Ruộng lúa mùa khác | [bai-a-phan-doan-ruong-lua/statement.md](bai-a-phan-doan-ruong-lua/statement.md) | CV (phân đoạn + đếm) | 0.7·mIoU + 0.3·(1 − min(1, MAE/3)) | 0.31 → 0.57 | 15 phút |
| B — Tín dụng vi mô qua thời gian | [bai-b-tin-dung-vi-mo/statement.md](bai-b-tin-dung-vi-mo/statement.md) | ML (dữ liệu bảng) | ROC-AUC (kèm Brier) | 0.68 → 0.79 | 10 phút |

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài: $100\cdot\operatorname{clip}\big((\text{metric} - \text{baseline}) / (\text{BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn** do BTC giữ (cùng định dạng và phân phối với `test_public`, không có đáp án).
- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** trọng số pretrained / LLM / API ngoài,
  **không** dùng đáp án của `test_public` để huấn luyện (chỉ để tự chấm).
- Nộp cho mỗi bài: code (huấn luyện + suy luận, chạy lại được trong giới hạn thời gian của bài), trọng số mô hình,
  file dự đoán `submission.csv`, báo cáo kỹ thuật ngắn (≤ 2 trang: tiền xử lý, mô hình, validation, kết quả).
- Gợi ý phân bổ thời gian: 30 phút đọc đề + chạy baseline cả hai bài; ~2.5 giờ mỗi bài; 30 phút cuối kiểm tra
  định dạng nộp, chạy lại pipeline từ đầu và viết báo cáo.

## Cách chạy

```bash
cd bai-a-phan-doan-ruong-lua   # hoặc bai-b-tin-dung-vi-mo
unzip public_data.zip          # tạo data/train và data/test_public
python baseline.py             # ghi submission.csv (đọc data/test_public)
python score.py                # tự chấm submission.csv trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (được dùng thêm PyTorch, nhưng không trọng số pretrained).
