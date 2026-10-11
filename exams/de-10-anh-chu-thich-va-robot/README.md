# Đề 10 — Ảnh–chú thích & Robot kho hàng

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, chủ đề
**ngôn ngữ gắn với thế giới**: câu tiếng Việt phải được hiểu thông qua những gì nhìn thấy (bài A) và những gì robot
làm (bài B). Độ khó nhỉnh hơn đề tham khảo: âm tính khó, dịch chuyển ánh sáng, tổ hợp chưa từng gặp, hành vi ẩn
khác nhau giữa các robot và nhãn có nhiễu.

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Tìm đúng bức ảnh | [bai-a-ghep-anh-chu-thich/statement.md](bai-a-ghep-anh-chu-thich/statement.md) | Đa phương thức (ảnh + tiếng Việt) | MRR trên 20 ứng viên (kèm R@1) | 0.237 → 0.99 | 15 phút |
| B — Sáu chú robot kho hàng | [bai-b-robot-kho-hang/statement.md](bai-b-robot-kho-hang/statement.md) | NLP + học hành vi | Độ chính xác trung bình theo robot | 0.604 → 0.96 | 10 phút |

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài: $100\cdot\operatorname{clip}\big((\text{điểm thô} - \text{baseline}) / (\text{Điểm BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn** do BTC giữ (cùng định dạng và kiểu phân phối với `test_public`, không có đáp án).
- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** trọng số pretrained / LLM / API ngoài,
  **không** dùng đáp án của `test_public` để huấn luyện (chỉ để tự chấm).
- Nộp cho mỗi bài: code (huấn luyện + suy luận, chạy lại được trong giới hạn thời gian của bài), trọng số mô hình
  (nếu có), file dự đoán `submission.csv`, báo cáo kỹ thuật ngắn (≤ 2 trang: tiền xử lý, mô hình, validation, kết quả).
- Gợi ý phân bổ thời gian: 30 phút đọc đề + chạy baseline cả hai bài; ~2.5 giờ mỗi bài; 30 phút cuối kiểm tra
  định dạng nộp, chạy lại pipeline từ đầu và viết báo cáo.

## Cách chạy

```bash
cd bai-a-ghep-anh-chu-thich    # hoặc bai-b-robot-kho-hang
unzip public_data.zip          # tạo data/train và data/test_public
python baseline.py             # ghi submission.csv (đọc data/test_public)
python score.py                # tự chấm submission.csv trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (được dùng thêm PyTorch, nhưng không trọng số pretrained).
