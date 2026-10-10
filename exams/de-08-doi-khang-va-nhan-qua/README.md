# Đề 08 — Đối kháng & Nhân quả

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, độ khó nhỉnh hơn
đề tham khảo. Bài A lấy cảm hứng từ bài *Double Agent Dilemma* của IOAI 2026: làm hai mô hình rất khác nhau bất đồng
bằng nhiễu nhỏ. Bài B là bài toán nhân quả: chọn khách để gửi voucher từ dữ liệu quan sát có nhiễu gây nhầm.

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Hai giám định viên | [bai-a-hai-mo-hinh-bat-dong/statement.md](bai-a-hai-mo-hinh-bat-dong/statement.md) | CV / ML an toàn (tấn công đối kháng MLP và rừng cây) | trung bình thành công × PF(‖δ‖₂) | 0.04 → 0.95 | 20 phút |
| B — Gửi voucher cho ai? | [bai-b-voucher-nhan-qua/statement.md](bai-b-voucher-nhan-qua/statement.md) | ML nhân quả (uplift, dữ liệu bảng) | uplift@30% (điểm phần trăm) | 2.5 → 14.6 | 10 phút |

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài: $100\cdot\operatorname{clip}\big((\text{metric} - \text{baseline}) / (\text{BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn** do BTC giữ (cùng định dạng và phân phối với `test_public`).
- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** trọng số pretrained / LLM / API ngoài,
  **không** dùng đáp án của `test_public` để huấn luyện (chỉ để tự chấm).
- Nộp cho mỗi bài: code (chạy lại được trong giới hạn thời gian của bài), file dự đoán (Bài A: `submission.npz`,
  Bài B: `submission.csv`), báo cáo kỹ thuật ngắn (≤ 2 trang: ý tưởng, cách kiểm chứng, kết quả).
- Riêng Bài B: BTC chạy lại mã nộp trong thư mục chỉ có `train.csv` và `test.csv` của tập ẩn; mã phải tự huấn luyện từ
  `train.csv`, không sao chép hay nhúng dữ liệu `test_public` (kể cả `answers.csv`) và không kèm tệp dữ liệu (CSV/NPY/pickle). Mã chạy
  bằng một lệnh duy nhất và ghi `submission.csv`; file dự đoán do BTC tạo khi chạy lại nên không cần nộp kèm.
- Gợi ý phân bổ thời gian: 30 phút đọc đề + chạy baseline cả hai bài; ~2.5 giờ mỗi bài; 30 phút cuối kiểm tra
  định dạng nộp, chạy lại pipeline từ đầu và viết báo cáo.

## Cách chạy

```bash
cd bai-a-hai-mo-hinh-bat-dong   # hoặc bai-b-voucher-nhan-qua
unzip public_data.zip           # tạo data/train và data/test_public (Bài A có thêm data/models.npz)
python baseline.py              # Bài A ghi submission.npz, Bài B ghi submission.csv (đọc data/test_public)
python score.py                 # tự chấm trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (được dùng thêm PyTorch, nhưng không trọng số pretrained).
Bài A kèm `models.py` — suy luận numpy của hai mô hình, cũng là phần lõi của `score.py`.
