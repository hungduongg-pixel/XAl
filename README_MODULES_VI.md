# Hệ thống chọn phương pháp XAI — bước 1 đến 4

Bộ code mở rộng từ `xai_selector.py` đã có, dành cho Python 3.10 trở lên.
Không cần cài thư viện ngoài để chạy quản lý phương pháp, bối cảnh, khảo sát và
đề xuất. Các câu lệnh bên dưới chạy trong thư mục sau khi giải nén.

## Chạy ví dụ hoàn chỉnh

```bash
python demo_pipeline.py
```

```text
Status: ready
LIME: DoTS=0.707031
SHAP_Waterfall: DoTS=0.644531
Selected: LIME
Excluded ALE (DoTS=0.791016): Không đáp ứng phạm vi giải thích yêu cầu: local.
```

Đây là **dữ liệu giả lập**. Ví dụ yêu cầu giải thích từng dự đoán (`local`), vì vậy
ALE bị loại dù DoTS cao nhất. Không diễn giải kết quả này thành kết luận về
chất lượng chung của LIME, SHAP hoặc ALE. Ví dụ cố ý giữ phản hồi ALE từ bộ dữ
liệu mẫu cũ để minh họa việc lọc kỹ thuật sau khi nhập dữ liệu đã thu thập.

Lưu các file ví dụ mới vào một thư mục:

```bash
python demo_pipeline.py --output-dir demo_output
```

Gói tải về cũng có sẵn `example_business_context.json`, `example_feedback.json`
và `example_recommendation.json` tương ứng.

## Các module và hàm chính

| File | Chức năng | API chính |
|---|---|---|
| `xai_methods.py` | Danh mục, đặc điểm và quy tắc tương thích | `list_methods`, `get_method`, `check_compatibility`, `compatible_methods` |
| `business_context.py` | Bối cảnh doanh nghiệp, mô hình, dữ liệu và người dùng | `ModelProfile`, `BusinessContext`, `collect_business_context` |
| `feedback.py` | Thu phản hồi theo từng giai đoạn, lưu/tải khảo sát | `FeedbackCollector`, `collect_feedback` |
| `recommendation.py` | Lọc kỹ thuật và xếp hạng DoTS | `recommend` |
| `xai_selector.py` | Chuẩn hóa và tính DoTS ban đầu | `evaluate`, `sensitivity` |
| `demo_pipeline.py` | Ví dụ tích hợp bằng dữ liệu giả lập | `build_demo` |

## 1. Danh mục và khả năng tương thích

| Phương pháp | Phạm vi trong danh mục | Dữ liệu | Điều kiện đáng chú ý |
|---|---|---|---|
| LIME | Local | Tabular; text/image classification | Phân loại cần xác suất lớp; dữ liệu bảng cần dữ liệu tham chiếu |
| SHAP | Local và global, tùy explainer/biểu đồ | Tabular, text, image | Cần chọn masker và prediction wrapper phù hợp |
| ALE | Global | Tabular, các biến numerical | Không dùng để giải thích từng dự đoán; bỏ các cột categorical khỏi tập biến cần giải thích |

Đối chiếu với tài liệu chính thức của
[LIME](https://lime-ml.readthedocs.io/en/latest/lime.html),
[SHAP Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html),
[TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
và [ALE trong Alibi](https://docs.seldon.ai/alibi-explain/explanations/methods/ale).

Quy tắc là đánh giá sơ bộ từ thông tin khai báo. Gói này chưa chạy explainer để
xác minh tương thích thực tế. Các điều kiện dữ liệu tham chiếu và đầu ra số là
chính sách thận trọng của luồng tích hợp này, không phải giới hạn tuyệt đối của
mọi triển khai XAI. Ví dụ, một số chế độ Tree SHAP có thể không cần dữ liệu nền.

`SHAP_Waterfall`, `SHAP_Decision`, `SHAP_Summary`, `SHAP_Dependence` được giữ làm
các lựa chọn riêng khi khảo sát; không gộp điểm đánh giá của các biểu đồ với nhau.
Waterfall/Decision được xét như giải thích cục bộ; Summary/Dependence là biểu đồ
tổng hợp trong danh mục này. `SHAP` là lựa chọn ở cấp họ phương pháp.

Dữ liệu chuỗi thời gian chưa có adapter trong danh mục. Nếu mô hình dự báo đã
chuyển đầu vào thành bảng đặc trưng trễ, khai báo `tabular` cho chính biểu diễn
mà bộ giải thích nhận, và bảo đảm bộ tiền xử lý nhất quán.

## 2. Tạo bối cảnh doanh nghiệp

```bash
python business_context.py --output my_context.json
```

Chương trình hỏi tên doanh nghiệp, lĩnh vực, mục tiêu AI, đối tượng sử dụng,
tên/phiên bản mô hình, họ mô hình, bài toán, loại dữ liệu, đầu ra mô hình, các
điều kiện tích hợp và phạm vi giải thích cần dùng. Các lựa chọn kỹ thuật dùng
các giá trị được hiển thị trong lời nhắc; câu hỏi yes/no nhập `yes` hoặc `no`.

Kiểm tra phương pháp phù hợp trước khi khảo sát:

```bash
python recommendation.py --context my_context.json
```

Khi chưa có khảo sát, hệ thống trả `feedback_required`, danh sách phương pháp
có thể đánh giá và lý do loại các phương pháp khác; không tạo điểm DoTS giả.

Tên lĩnh vực và trình độ người dùng được lưu làm bối cảnh, không tự chuyển thành
điểm ưu tiên chủ quan. Yêu cầu phạm vi giải thích là điều kiện lọc; các ưu tiên
và trải nghiệm của người đánh giá được đưa vào qua trọng số và phản hồi thực tế.

## 3. Thu Trust Before, Trust After và Satisfaction

```bash
python feedback.py --context my_context.json --participants 3 --output my_survey.json
```

Mặc định, lọc LIME/SHAP/ALE trước khi hỏi. Có thể chỉ định một bộ phương pháp:

```bash
python feedback.py --context my_context.json --methods LIME SHAP_Waterfall --participants 3 --output my_survey.json
```

Với mỗi người, nhập mã người đánh giá, `expertise_weight` thuộc (0, 1],
`trust_weight` thuộc [0, 1], rồi các bộ điểm đã chấm. Mỗi bộ mặc định có 8 điểm
từ 1 đến 5, ngăn cách bằng dấu cách, ví dụ `3 3 4 4 3 4 4 3`.

Luồng nhập:

1. Giới thiệu mô hình và kết quả dự đoán; thu **Trust Before**.
2. Trình bày giải thích của phương pháp thứ nhất; thu **Trust After** và
   **Satisfaction** cho phương pháp đó.
3. Tiếp tục cho các phương pháp còn lại, giữ cùng bộ câu hỏi và thứ tự câu.

Module cung cấp vị trí nhập điểm; không tự tạo bảng hỏi tâm lý được kiểm định.
Cần dùng công cụ khảo sát phù hợp với nghiên cứu và xử lý câu đảo chiều theo
hướng dẫn chấm điểm trước khi nhập. Cũng không tự trình bày hình giải thích:
người tổ chức khảo sát phải chuẩn bị và cho người đánh giá xem chúng.

Các kiểm soát dữ liệu:

- Phải đăng ký Trust Before trước khi nhập phản hồi sau giải thích.
- Không âm thầm ghi đè baseline hay phản hồi đã nộp.
- Không xuất DoTS khi một người chưa đánh giá đủ bộ phương pháp.
- Điểm ngoài miền, sai số lượng câu và trọng số không hợp lệ đều báo lỗi.
- Bối cảnh và phiên bản mô hình được gắn bằng `context_id`; đổi bối cảnh sẽ
  phát hiện phản hồi không khớp. Đây là kiểm tra nhất quán, không phải chữ ký số.

Lưu khảo sát đang làm dở và tiếp tục bằng API:

```python
from feedback import FeedbackCollector

survey = FeedbackCollector(["LIME", "SHAP"])
survey.register_participant("R1", [3] * 8, expertise_weight=1, trust_weight=0.5)
survey.submit_feedback("R1", "LIME", [4] * 8, [4] * 8)
survey.save("progress.json", allow_incomplete=True)

resumed = FeedbackCollector.load("progress.json")
print(resumed.missing_feedback())   # {'R1': ['SHAP']}
resumed.submit_feedback("R1", "SHAP", [5] * 8, [4] * 8)
resumed.save("survey.json")
```

## 4. Đề xuất dựa trên kỹ thuật và DoTS

```bash
python recommendation.py --context my_context.json --feedback my_survey.json --output result.json
```

Hoặc chạy ngay với các file có sẵn:

```bash
python recommendation.py --context example_business_context.json --feedback example_feedback.json --output result.json
```

Chính sách kết hợp: **lọc điều kiện kỹ thuật → xếp hạng theo DoTS nguyên gốc**.
Không cộng điểm kỹ thuật tùy ý vào DoTS. `recommendation_score` bằng DoTS đối với
phương pháp hợp lệ đã được đánh giá. Chính sách lọc là phần mở rộng của bộ code,
không phải công thức mới được gán cho bài báo.

File kết quả ghi:

- `ranking` và `best_methods`: các phương pháp phù hợp đã đánh giá; giữ đồng điểm.
- `excluded_methods`: phương pháp bị loại và lý do cụ thể, kể cả khi DoTS cao.
- `awaiting_feedback`: phương pháp phù hợp chưa được đánh giá; không gán 0 điểm.
- `dots_report`: điểm nguyên gốc và các thành phần của từng người.
- `status`: `ready`, `feedback_required`, `partial_feedback`, hoặc `no_compatible_method`.

Nếu muốn xem thêm một phương pháp chưa có trong khảo sát, truyền bộ ứng viên rộng
hơn qua `method_ids`. Đề xuất khi ấy là tạm thời (`partial_feedback`). Khảo sát
thiếu câu trả lời của người đã tham gia sẽ báo lỗi; chương trình không tự loại
người đó hoặc đổi tập người chấm giữa các phương pháp.

```python
from business_context import BusinessContext
from feedback import FeedbackCollector
from recommendation import recommend

context = BusinessContext.load("example_business_context.json")
feedback = FeedbackCollector.load("example_feedback.json")
report = recommend(context, feedback.to_experiment())
print(report["best_methods"])
```

Với JSON cũ chưa có `context_id`, vẫn có thể nhập nhưng báo chú thích yêu cầu
kiểm tra bối cảnh. Các dữ liệu mẫu đi kèm đã được gắn nhãn `synthetic_demo`.

## Kiểm tra và phạm vi

```bash
python -m unittest -v test_xai_selector.py test_modules.py
```

Đã kiểm tra 40 trường hợp, bao gồm thuật toán DoTS, khả năng tương thích, thu thập
theo giai đoạn, lưu/tải và đề xuất toàn luồng. Toàn bộ code có thể chạy độc lập
với các gói LIME/SHAP/Alibi vì chưa sinh giải thích trong phiên bản này.

Không triển khai `app.py` hay giao diện Streamlit theo phạm vi bạn yêu cầu.
Nếu nối với mô hình thật sau này, cần cấu hình prediction wrapper, encoding,
masker, lựa chọn đầu ra/lớp cần giải thích và kiểm tra phiên bản thư viện thực tế.
