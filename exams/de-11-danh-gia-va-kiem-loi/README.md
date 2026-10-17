# Đề 11 — Đánh giá khách sạn & Kiểm lỗi gạch men

Đề thi OlpAI (Olympic Trí tuệ nhân tạo Sinh viên Việt Nam), trình bày theo khung IOAI 2026, gồm một bài **xử lý ngôn
ngữ tự nhiên** và một bài **thị giác máy tính**. Bài A đọc đánh giá của khách lưu trú để biết họ khen, chê điều gì. Bài B
soi ảnh bề mặt gạch men trên dây chuyền để phát hiện viên lỗi và chỉ ra vị trí lỗi. Cả hai bài đều có dịch chuyển phân
phối giữa dữ liệu huấn luyện và dữ liệu chấm. Muốn điểm cao, đội phải khám phá dữ liệu kỹ và validation đúng cách.

| Bài | Đề chính thức | Lĩnh vực | Metric | Baseline → 100 điểm | Thời gian chạy |
|---|---|---|---|---|---|
| A — Khen chê từng khía cạnh | [bai-a-cam-xuc-khia-canh/statement.md](bai-a-cam-xuc-khia-canh/statement.md) | NLP (cảm xúc theo khía cạnh, tiếng Việt) | micro-F1 trên bộ ba (đánh giá, khía cạnh, cực tính) | 0.725 → 0.97 | 10 phút |
| B — Kiểm lỗi gạch men | [bai-b-kiem-loi-gach-men/statement.md](bai-b-kiem-loi-gach-men/statement.md) | Thị giác máy tính (phát hiện + chỉ vị trí bất thường) | 0.7·AUROC + 0.3·Pointing | 0.413 → 0.795 | 15 phút |

## Quy chế chung

- Thi theo đội 3 người, **6 giờ liên tục**, 2 bài, mỗi bài 100 điểm, tổng 200 điểm.
- Điểm mỗi bài: $100\cdot\operatorname{clip}\big((\text{điểm thô} - \text{baseline}) / (\text{Điểm BTC} - \text{baseline}), 0, 1\big)$,
  chấm trên **tập ẩn** do BTC giữ (cùng định dạng và kiểu phân phối với `test_public`, không có đáp án).
- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** trọng số pretrained / LLM / API ngoài,
  **không** dùng đáp án của `test_public` để huấn luyện (chỉ để tự chấm), **không** gán nhãn thủ công hay sửa tay dự đoán
  trên dữ liệu test. Được dùng dữ liệu test không nhãn theo bất kỳ cách nào (chuẩn hoá, học không giám sát…).
- Hình thức lấy dự đoán trên tập ẩn do BTC thông báo đầu buổi thi: BTC phát phần input của tập ẩn để đội tự chạy và nộp
  file dự đoán, hoặc BTC chạy lại pipeline của đội. Vì vậy code phải đọc được dữ liệu test cùng định dạng `test_public` và
  chạy lại được từ đầu trong giới hạn thời gian của bài.
- Nộp cho mỗi bài: code (huấn luyện + suy luận), file dự đoán `submission.csv`, báo cáo kỹ thuật ngắn (≤ 2 trang: tiền xử lý,
  mô hình, validation, kết quả).
- Gợi ý phân bổ thời gian: 30 phút đọc đề + chạy baseline cả hai bài; ~2.5 giờ mỗi bài; 30 phút cuối kiểm tra
  định dạng nộp, chạy lại pipeline từ đầu và viết báo cáo.

## Cách chạy

```bash
cd bai-a-cam-xuc-khia-canh     # hoặc bai-b-kiem-loi-gach-men
unzip public_data.zip          # tạo data/train và data/test_public
python baseline.py             # ghi submission.csv (đọc data/test_public)
python score.py                # tự chấm submission.csv trên test_public
```

Các cờ chung: `baseline.py --data DIR --out FILE --split {public,private}`,
`score.py --data DIR --pred FILE --split {public,private}`.
Môi trường: Python 3.11, numpy, pandas, scikit-learn, scipy, pillow (được dùng thêm PyTorch, nhưng không trọng số pretrained).
