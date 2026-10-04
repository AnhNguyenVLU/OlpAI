# Đề 01 — Dịch máy & Nhận dạng ngôn ngữ ký hiệu

Đề luyện thi OlpAI (mô phỏng đề tham khảo OlpAI'25, trình bày theo khung IOAI 2026), độ khó nhỉnh hơn
đề tham khảo: có dịch chuyển phân phối (câu dài hơn, tổ hợp mới, người ký mới, thuận tay trái), nhiễu nhãn
và một số bẫy chuẩn hoá dữ liệu.

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Phiên dịch tiếng Kơ Ru | [bai-a-dich-may/statement.md](bai-a-dich-may/statement.md) | NLP (dịch máy) | chrF (β=1, n=6) toàn corpus | 44.1 → 95.0 | 10 phút |
| B — Đôi tay biết nói | [bai-b-ky-hieu/statement.md](bai-b-ky-hieu/statement.md) | CV / chuỗi thời gian | macro-F1 (30 lớp) | 0.583 → 0.94 | 10 phút |

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài: $100\cdot\operatorname{clip}\big((\text{metric} - \text{baseline}) / (\text{BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn** do BTC giữ (cùng định dạng và phân phối với `test_public`, không có đáp án).
- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** trọng số pretrained / LLM / API ngoài,
  **không** dùng đáp án `test_public/answers.csv` để huấn luyện (chỉ để tự chấm), không dùng đầu vào test để huấn luyện.
- Nộp cho mỗi bài: code (huấn luyện + suy luận, chạy lại được trong 10 phút), trọng số mô hình, file dự đoán
  `submission.csv`, báo cáo kỹ thuật ngắn (≤ 2 trang: tiền xử lý, mô hình, validation, kết quả).
- Gợi ý phân bổ thời gian: 30 phút đọc đề + chạy baseline cả hai bài; ~2.5 giờ mỗi bài; 30 phút cuối kiểm tra
  định dạng nộp, chạy lại pipeline từ đầu và viết báo cáo.

## Cách chạy

```bash
cd bai-a-dich-may            # hoặc bai-b-ky-hieu
unzip public_data.zip        # tạo data/train và data/test_public
python baseline.py           # ghi submission.csv (đọc data/test_public)
python score.py --pred submission.csv          # tự chấm trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
