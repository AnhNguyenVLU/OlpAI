# Đề 06 — Tiếng chim & Hội thoại

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, độ khó nhỉnh hơn
đề tham khảo: đổi thiết bị thu (máy ghi tự động thay máy cầm tay), nhiễu nhãn, tiếng chim chồng lấn trong tạp âm rừng; hội
thoại chăm sóc khách hàng bị xáo trộn, không nhãn người nói, nghiệp vụ và văn phong mới ở tập test (bài B lấy cảm hứng từ
bài *Find the Order* của IOAI 2026).

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Lắng nghe Cúc Phương | [bai-a-tieng-chim/statement.md](bai-a-tieng-chim/statement.md) | Âm thanh / ML (nhận dạng đa nhãn) | macro-AP theo loài (kèm micro-AP) | 0.19 → 0.78 | 20 phút |
| B — Tổng đài bị xáo trộn | [bai-b-sap-xep-hoi-thoai/statement.md](bai-b-sap-xep-hoi-thoai/statement.md) | NLP (khôi phục thứ tự chuỗi) | độ chính xác thứ tự theo cặp (1 − I/M) | 0.833 → 0.926 | 15 phút |

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài: $100\cdot\operatorname{clip}\big((\text{metric} - \text{baseline}) / (\text{BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn** do BTC giữ (cùng định dạng và phân phối với `test_public`, không có đáp án).
- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** trọng số pretrained / LLM / API ngoài,
  **không** dùng đáp án của `test_public` để huấn luyện (chỉ để tự chấm), kể cả sao chép hoặc nhúng dữ liệu/trọng số lấy từ
  `test_public` vào mã nộp.
- Nộp cho mỗi bài: mã nguồn (huấn luyện + suy luận, chạy lại được trong giới hạn thời gian của bài) và báo cáo kỹ thuật ngắn
  (≤ 2 trang: tiền xử lý, mô hình, validation, kết quả).
- Khi chấm, BTC chạy lại toàn bộ pipeline (huấn luyện + suy luận) từ mã nộp trong thư mục chỉ có `data/train/` và tập chấm ẩn
  (đặt ở `data/test_public/`, không có đáp án): mã phải tự huấn luyện từ `train/`, không kèm trọng số hay dữ liệu tính sẵn, chạy bằng một
  lệnh duy nhất (ví dụ `python solution.py`) và ghi `submission.csv`. File dự đoán do BTC tạo khi chạy lại nên không cần nộp kèm.
  BTC rà soát mã của các đội đứng đầu.
- Gợi ý phân bổ thời gian: 30 phút đọc đề + chạy baseline cả hai bài; ~2.5 giờ mỗi bài; 30 phút cuối kiểm tra
  định dạng nộp, chạy lại pipeline từ đầu và viết báo cáo.

## Cách chạy

```bash
cd bai-a-tieng-chim            # hoặc bai-b-sap-xep-hoi-thoai
unzip public_data.zip          # tạo data/train và data/test_public
python baseline.py             # ghi submission.csv (đọc data/test_public)
python score.py                # tự chấm submission.csv trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (được dùng thêm PyTorch, nhưng không trọng số pretrained).
Bài A đọc WAV bằng `scipy.io.wavfile`.
