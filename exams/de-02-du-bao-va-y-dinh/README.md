# Đề 02 — Dự báo phụ tải & Ý định đa nhãn

Đề luyện thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026.
Thi theo đội 3 người, **6 giờ liên tục**, 2 bài độc lập. Mỗi bài chấm thang 0–100, tổng tối đa 200.

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy | Bẫy chính |
|---|---|---|---|---|---|---|
| A | [Tuần lễ của điều độ viên](bai-a-du-bao-phu-tai/statement.md) | ML — chuỗi thời gian, dự báo xác suất | pinball loss P10/P50/P90 chuẩn hoá theo trạm (↓) | 0.141 → 0.045 | 10 phút | ngày lễ & làm bù, sự cố trong lịch sử, cold-start, dự báo nhiệt độ nhiễu, điểm gãy, độ bất định khác nhau |
| B | [Tổng đài viên không ngủ](bai-b-y-dinh-da-nhan/statement.md) | NLP — phân loại đa nhãn tiếng Việt | ½(micro-F1 + macro-F1) (↑) | 0.755 → 0.875 | 10 phút | train có dấu ↔ test không dấu/teencode/Telex, diễn đạt mới, nhiễu nhãn 8%, nhãn hiếm |

Gợi ý phân bổ thời gian: ~30 phút đọc đề + khám phá dữ liệu cả hai bài, sau đó chia đội (Bài A
thường cần nhiều công feature engineering + backtest hơn; Bài B cần chuẩn hoá văn bản + chọn ngưỡng),
dành 45 phút cuối để chạy lại pipeline sạch, kiểm tra định dạng nộp và viết báo cáo.

## Quy chế chung

- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** pretrained weights, **không** LLM/API ngoài.
- **Không** dùng đáp án `test_public` để huấn luyện — chỉ để tự chấm.
- Tập chấm ẩn do BTC giữ, cùng định dạng `test_public`, không có đáp án. Lúc chấm, `data/test_public/`
  được thay bằng tập ẩn; pipeline phải chạy lại trọn vẹn trong giới hạn thời gian của từng bài.
- Nộp cho mỗi bài: code (train + inference), trọng số mô hình (nếu có), file dự đoán, báo cáo kỹ thuật ngắn (≤ 2 trang).
- Điểm quy đổi: `100 · clip((điểm thô − baseline) / (Điểm BTC − baseline), 0, 1)` — xem `score.py` của từng bài
  (Bài A metric giảm là tốt).

## Cách chạy

```bash
cd bai-a-du-bao-phu-tai               # hoặc bai-b-y-dinh-da-nhan
unzip public_data.zip                 # tạo data/train/ và data/test_public/
python baseline.py                    # ghi submission.csv (mặc định đọc data/test_public)
python score.py --pred submission.csv # tự chấm trên test_public
```

Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (thí sinh được dùng thêm PyTorch... nhưng không pretrained).
