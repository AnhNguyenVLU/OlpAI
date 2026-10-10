# Đề 09 — Biển số & Địa chỉ (chủ đề "giao hàng thông minh")

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, độ khó nhỉnh hơn
đề tham khảo: dịch chuyển miền về phông chữ và điều kiện chụp, mã tỉnh chưa từng gặp, địa chỉ ghi theo đơn vị hành chính
cũ/mới lẫn lộn, xã bị chia theo tuyến đường, dữ liệu gõ tay đủ kiểu (không dấu, viết tắt, sai chính tả, Unicode tổ hợp).

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Camera trạm thu phí: đọc biển số | [bai-a-doc-bien-so/statement.md](bai-a-doc-bien-so/statement.md) | CV (OCR) | trung bình 1 − min(1, Levenshtein / độ dài) (kèm exact match) | 0.44 → 0.95 | 20 phút |
| B — Shipper đọc địa chỉ sau sáp nhập | [bai-b-chuan-hoa-dia-chi/statement.md](bai-b-chuan-hoa-dia-chi/statement.md) | NLP (chuẩn hoá, khớp danh mục) | độ chính xác trung bình 4 trường (kèm đúng cả địa chỉ) | 0.930 → 0.995 | 10 phút |

Mọi dữ liệu đều là **tổng hợp**. Biển số theo phong cách Việt Nam ở dạng đơn giản hoá; đơn vị hành chính ở Bài B là
**hư cấu**.

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài: $100\cdot\operatorname{clip}\big((\text{metric} - \text{baseline}) / (\text{BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn** do BTC giữ (cùng định dạng và phân phối với `test_public`, không có đáp án).
- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** trọng số pretrained / LLM / API ngoài,
  **không** dùng đáp án của `test_public` để huấn luyện (chỉ để tự chấm).
- Nộp cho mỗi bài: code (huấn luyện + suy luận, chạy lại được trong giới hạn thời gian của bài), trọng số mô hình (nếu có),
  file dự đoán `submission.csv`, báo cáo kỹ thuật ngắn (≤ 2 trang: tiền xử lý, mô hình, validation, kết quả).
- Gợi ý phân bổ thời gian: 30 phút đọc đề + chạy baseline cả hai bài; ~2.5 giờ mỗi bài; 30 phút cuối kiểm tra
  định dạng nộp, chạy lại pipeline từ đầu và viết báo cáo.

## Cách chạy

```bash
cd bai-a-doc-bien-so           # hoặc bai-b-chuan-hoa-dia-chi
unzip public_data.zip          # tạo data/train, data/test_public (Bài B có thêm data/gazetteer)
python baseline.py             # ghi submission.csv (đọc data/test_public)
python score.py                # tự chấm submission.csv trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (được dùng thêm PyTorch, nhưng không trọng số pretrained).
