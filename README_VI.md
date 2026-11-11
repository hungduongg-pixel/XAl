# Python chọn phương pháp XAI theo DoTS

**Bản mở rộng bước 1–4:** đã bổ sung `xai_methods.py`, `business_context.py`,
`feedback.py` và `recommendation.py`. Xem [hướng dẫn các module](README_MODULES_VI.md)
để chạy luồng tích hợp bằng `python demo_pipeline.py`. Các phần bên dưới mô tả
riêng bộ tính DoTS ban đầu. Không có bước giao diện Streamlit trong gói này.

Bản triển khai độc lập phần tính điểm và lựa chọn XAI của Reis và cộng sự,
*Computers in Industry* 165 (2025), 104233.

- Bài báo: https://doi.org/10.1016/j.compind.2024.104233
- Toàn văn trên trang tác giả: https://www.di.uminho.pt/~jmf/PUBLI/papers/2025-compind.pdf
- Mã nguồn XAISelector do tác giả công bố: https://github.com/mindior/xaiselector

Đây là phần lõi tính DoTS theo công thức (1), (2), (4) và Algorithm 1.
Chương trình nhận phản hồi của người đánh giá về những giải thích XAI đã có.
Nó không huấn luyện mô hình, sinh giải thích SHAP/LIME/ALE hay thay thế toàn bộ
ứng dụng khảo sát Flask của tác giả. Dữ liệu kèm theo hoàn toàn là minh họa;
không tái lập kết quả thực nghiệm của bài báo.

## 1. Chạy ngay

Cần Python 3.10 trở lên, **không cần cài thư viện ngoài**. Giải nén gói và mở
cửa sổ lệnh trong thư mục chứa các file:

```bash
python xai_selector.py
```

Kết quả dữ liệu minh họa, khi `trust_weight = 0.5`:

```text
Method                        DoTS    Trust(wE)  Satisf.(wE)
ALE                       0.791016     0.609375     0.972656
LIME                      0.707031     0.648438     0.765625
SHAP_Waterfall             0.644531     0.750000     0.539062
Selected: ALE
```

Tên phương pháp ở đây chỉ là nhãn gắn với các câu trả lời giả lập;
bảng trên không khẳng định ALE luôn tốt hơn LIME hoặc SHAP.

## 2. Cách tính

Với thang 1–5, mỗi người đánh giá có:

```text
T = trung bình ((trust_after - trust_before + 4) / 8)
S = trung bình ((satisfaction - 1) / 4)
score = w_I * T + (1 - w_I) * S
DoTS = tổng (w_E * score) / tổng w_E
```

`w_E` là `expertise_weight`, thuộc (0, 1]; `w_I` là `trust_weight`, thuộc [0, 1].
Tăng `w_I` nghĩa là ưu tiên thành phần tin cậy. Khi tin cậy trước và sau bằng nhau,
**T = 0.5**, không phải 0. Chọn phương pháp có DoTS lớn nhất; giữ các phương pháp
đồng điểm trong sai số 1e-12. DoTS không phải xác suất dự đoán đúng.

## 3. Thay bằng dữ liệu của bạn

Mở `example_survey.json`. Mỗi người có cấu trúc sau; các dấu `...` chỉ minh họa
ở tài liệu này, không được đưa vào JSON thật:

```text
{
  "id": "R1",
  "expertise_weight": 1.0,
  "trust_weight": 0.5,
  "trust_before": [3, 3, 3, 3, 3, 3, 3, 3],
  "responses": {
    "LIME": {
      "trust_after": [4, 4, 4, 5, 4, 4, 4, 3],
      "satisfaction": [4, 4, 4, 4, 5, 4, 4, 3]
    },
    ...
  }
}
```

Thu thập `trust_before` sau khi giới thiệu mô hình, trước khi trình bày giải thích.
Thu thập `trust_after` và `satisfaction` sau khi người dùng xem từng giải thích.

- Mặc định mỗi thang có 8 câu, miền giá trị 1–5. Có thể đổi `min`, `max`, `items`
  trong `trust_scale` và `satisfaction_scale`.
- Giữ cùng bộ câu hỏi, cùng thứ tự câu hỏi, cùng tập người đánh giá và cùng bộ
  phương pháp trong một lần so sánh.
- Đầu vào phải được chấm điểm cùng chiều: giá trị lớn tương ứng tin cậy/hài lòng
  cao. Nếu bảng hỏi có câu đảo chiều, xử lý theo hướng dẫn chấm điểm của thang đo
  trước khi nhập. Chương trình không tự suy ra câu đảo chiều.
- Không tự thay câu trả lời thiếu bằng 0 hoặc tự bỏ người đánh giá. Dữ liệu không
  hợp lệ sẽ báo lỗi để bạn sửa tại nguồn.
- Chọn `expertise_weight` theo quy tắc thống nhất của nghiên cứu; chương trình
  nhận trọng số trực tiếp, không tự quy đổi số năm kinh nghiệm.
- Đổi `data_kind` từ `synthetic_demo` sang `real_survey` khi dùng dữ liệu thật.

Chạy khảo sát và lưu kết quả:

```bash
python xai_selector.py --input example_survey.json --output report.json
```

`report.json` chứa `ranking`, `best_methods` và `per_respondent` để kiểm tra
từng thành phần. `trust_weighted` và `satisfaction_weighted` là thống kê mô tả
theo trọng số chuyên môn; DoTS luôn được tính từ điểm phối hợp của từng người.

## 4. Phân tích độ nhạy

Thử cùng một trọng số tin cậy cho tất cả người đánh giá:

```bash
python xai_selector.py --input example_survey.json --trust-weight 0.8
python xai_selector.py --input example_survey.json --sensitivity sensitivity.json
```

Lệnh thứ hai quét từ 0 đến 1 với bước 0.025, lưu điểm và phương pháp đứng đầu
tại từng mức. Với dữ liệu minh họa, ưu tiên hoàn toàn sự hài lòng chọn ALE;
ưu tiên hoàn toàn tin cậy chọn SHAP_Waterfall. Đây là minh họa việc lựa chọn
phụ thuộc ưu tiên của người đánh giá.

## 5. Dùng như thư viện

```python
import json
from pathlib import Path
from xai_selector import evaluate, sensitivity

survey = json.loads(Path("example_survey.json").read_text(encoding="utf-8"))
report = evaluate(survey)
print(report["best_methods"])
print(report["ranking"])

grid = sensitivity(survey, weights=[0, 0.25, 0.5, 0.75, 1])
```

## 6. Kiểm tra

```bash
python -m unittest -v test_xai_selector.py
```

Các kiểm tra bao gồm biên 0/1, trường hợp không đổi mức tin cậy, ví dụ trọng số
chuyên môn trong bài, trọng số tin cậy khác nhau giữa từng người, đồng điểm,
đổi thang đo, câu trả lời thiếu, trọng số sai và kết quả minh họa tính tay.

Phần đánh giá độ đồng thuận giữa người chấm (weighted kappa), kiểm tra độ nhất
quán thang đo và giao diện khảo sát chưa được triển khai trong gói này.
