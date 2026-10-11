# Đề 07 — Xe máy giờ cao điểm & Bóng ma trong kho lưu trữ

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, độ khó nhỉnh hơn
đề tham khảo: dịch chuyển miền (ban đêm, mưa, camera mới; thể loại và đề tài mới), vật gây nhiễu, che khuất, ranh giới
mơ hồ giữa người và máy, và những tín hiệu "thấy rõ trong train" nhưng không còn đúng ở test.

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Giờ cao điểm | [bai-a-phat-hien-xe-may/statement.md](bai-a-phat-hien-xe-may/statement.md) | CV (phát hiện đối tượng) | AP@0.5 một lớp, nội suy kiểu PASCAL VOC (kèm AP@0.75) | 0.07 → 0.79 | 45 phút |
| B — Bóng ma trong kho lưu trữ | [bai-b-ranh-gioi-van-ban-may/statement.md](bai-b-ranh-gioi-van-ban-may/statement.md) | NLP (tìm ranh giới văn bản người/máy) | trung bình $\exp(-\lvert p-t\rvert/100)$ theo ký tự | 0.26 → 0.88 | 10 phút |

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
cd bai-a-phat-hien-xe-may      # hoặc bai-b-ranh-gioi-van-ban-may
unzip public_data.zip          # tạo data/train và data/test_public
python baseline.py             # ghi submission.csv (đọc data/test_public)
python score.py                # tự chấm submission.csv trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (được dùng thêm PyTorch, nhưng không trọng số pretrained).
