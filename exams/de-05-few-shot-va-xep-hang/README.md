# Đề 05 — Few-shot & Xếp hạng

Đề luyện thi OlpAI (format bài theo IOAI 2026). Thi theo đội 3 người, **6 giờ liên tục**, 2 bài độc lập.

| Bài | Lĩnh vực | Nhiệm vụ | Metric (điểm chính) | Baseline → 100 điểm | Giới hạn chạy | Đề chi tiết |
|---|---|---|---|---|---|---|
| A — Đọc chữ Nôm của người chép mới | CV | Phân loại few-shot 20-way 5-shot trên lớp chữ **mới hoàn toàn** | Accuracy trung bình theo episode | 0.642 → 0.775 | 10 phút / 4 CPU | [statement.md](bai-a-chu-nom-few-shot/statement.md) |
| B — Tuần sau học gì? | ML (learning-to-rank) | Xếp hạng top-10 khoá học người dùng sẽ ghi danh tuần tới | NDCG@10 (báo thêm Recall@10) | 0.034 → 0.282 | 15 phút / 4 CPU | [statement.md](bai-b-goi-y-khoa-hoc/statement.md) |

**Thang điểm:** mỗi bài 100 điểm, tổng 200. Điểm mỗi bài quy từ điểm thô trên **tập chấm ẩn**:
`Điểm = 100 · clip((điểm thô − baseline) / (Điểm BTC − baseline), 0, 1)` — baseline.py được 0 điểm, lời giải
tham chiếu của BTC (Điểm BTC) được 100 điểm (ngưỡng ghi trong từng `statement.md` và `score.py`).

**Gợi ý phân bổ thời gian:** 30 phút đọc đề + dựng validation đúng cách cho cả hai bài; ~2.5 giờ mỗi bài;
30 phút cuối đóng gói code, trọng số, file dự đoán và báo cáo.

## Quy chế chung

- Chỉ dùng dữ liệu BTC cấp. **Không** dữ liệu ngoài, **không** pretrained weights / LLM / API ngoài.
- **Không** dùng tập test (kể cả `answers.csv` của `test_public`) để huấn luyện — nó chỉ để tự chấm.
- Tập chấm ẩn do BTC giữ, cùng định dạng `test_public`, không có đáp án.
- Bài nộp mỗi bài: code (huấn luyện + suy luận chạy lại được), model weights, file dự đoán
  `submission.csv` trên tập được giao, báo cáo kỹ thuật ngắn (≤ 2 trang: tiền xử lý, mô hình, validation, kết quả).
- Đội được dùng thư viện tuỳ ý (numpy, pandas, scikit-learn, PyTorch…) miễn tuân thủ giới hạn chạy.

## Cách chạy (thí sinh)

```bash
cd bai-a-chu-nom-few-shot        # hoặc bai-b-goi-y-khoa-hoc
unzip public_data.zip            # tạo data/train/ và data/test_public/
python baseline.py               # ghi submission.csv cho test_public
python score.py --pred submission.csv   # tự chấm trên test_public (in JSON)
```
