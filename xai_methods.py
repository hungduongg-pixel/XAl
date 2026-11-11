"""Method catalogue and conservative, metadata-based compatibility rules.

These are preflight rules, not runtime verification of an installed explainer.
No explainer is executed by this module. Sources are attached to each family.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from business_context import BusinessContext, MODEL_TYPES


@dataclass(frozen=True)
class XAIMethod:
    method_id: str
    family: str
    description: str
    scopes: tuple[str, ...]
    data_types: tuple[str, ...]
    task_types: tuple[str, ...]
    model_types: tuple[str, ...]
    model_agnostic: bool
    limitations: tuple[str, ...]
    sources: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


_BASE_METHODS = (
    XAIMethod(
        "LIME", "LIME", "Xấp xỉ cục bộ để giải thích một dự đoán.",
        ("local",), ("tabular", "text", "image"), ("classification", "regression"),
        MODEL_TYPES, True,
        ("Bộ giải thích text/image tiêu chuẩn trong danh mục này chỉ hỗ trợ classification.",
         "Kết quả phụ thuộc lấy mẫu lân cận; cần kiểm tra độ ổn định."),
        ("https://lime-ml.readthedocs.io/en/latest/lime.html",),
    ),
    XAIMethod(
        "SHAP", "SHAP", "Phân bổ đóng góp đặc trưng; có nhiều bộ giải thích và kiểu biểu đồ.",
        ("local", "global"), ("tabular", "text", "image"), ("classification", "regression"),
        MODEL_TYPES, True,
        ("Phải chọn explainer và masker phù hợp; hỗ trợ mô hình cụ thể cần kiểm tra khi chạy.",
         "Giải thích tổng thể cần tổng hợp trên tập mẫu đại diện."),
        ("https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html",
         "https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html"),
    ),
    XAIMethod(
        "ALE", "ALE", "Ước lượng ảnh hưởng tổng thể của biến số trên phân phối dữ liệu.",
        ("global",), ("tabular",), ("classification", "regression"),
        MODEL_TYPES, True,
        ("Phạm vi triển khai tham chiếu: ALE số của Alibi; không giải thích biến categorical.",
         "Không thay thế giải thích cho từng dự đoán."),
        ("https://docs.seldon.ai/alibi-explain/explanations/methods/ale",),
    ),
)
_REGISTRY = {method.method_id: method for method in _BASE_METHODS}
# Retain variants as distinct survey treatments; do not merge them into SHAP.
for _variant, _scopes in {
    "SHAP_Waterfall": ("local",), "SHAP_Decision": ("local",),
    "SHAP_Summary": ("global",), "SHAP_Dependence": ("global",),
}.items():
    _REGISTRY[_variant] = replace(_REGISTRY["SHAP"], method_id=_variant, scopes=_scopes)


def get_method(method_id: str) -> XAIMethod:
    if not isinstance(method_id, str):
        raise ValueError("method_id: expected a string")
    for name, method in _REGISTRY.items():
        if name.casefold() == method_id.strip().casefold():
            return method
    raise ValueError(f"Unknown XAI method: {method_id}")


def normalize_methods(method_ids):
    if not isinstance(method_ids, (list, tuple)) or not method_ids:
        raise ValueError("method_ids: expected a nonempty list/tuple")
    names = tuple(get_method(name).method_id for name in method_ids)
    if len(set(names)) != len(names):
        raise ValueError("method_ids: duplicate methods/aliases")
    return names


def list_methods(include_variants=False):
    return tuple(_REGISTRY.values()) if include_variants else _BASE_METHODS


@dataclass(frozen=True)
class CompatibilityResult:
    method_id: str
    eligible: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    suggested_adapter: str | None

    def to_dict(self):
        return asdict(self)


def check_compatibility(method_id: str, context: BusinessContext) -> CompatibilityResult:
    if not isinstance(context, BusinessContext):
        raise ValueError("context: expected a BusinessContext")
    method = get_method(method_id)
    model = context.model
    reasons, notes = [], list(method.limitations)
    adapter = None
    if model.data_type not in method.data_types:
        reasons.append(f"Không hỗ trợ dữ liệu {model.data_type} trong danh mục hiện tại.")
    if model.task_type not in method.task_types:
        reasons.append(f"Không hỗ trợ bài toán {model.task_type}.")
    requested = {"local", "global"} if context.required_scope == "both" else {context.required_scope}
    if context.required_scope != "any" and not requested.issubset(method.scopes):
        reasons.append(f"Không đáp ứng phạm vi giải thích yêu cầu: {context.required_scope}.")
    if not model.has_prediction_function:
        reasons.append("Thiếu hàm dự đoán cho nhiều mẫu.")
    if not model.preprocessing_ready:
        reasons.append("Cần chuẩn bị bộ tiền xử lý cho đầu vào của explainer.")
    # Conservative workflow: all tabular routes use representative reference data,
    # including SHAP fallback routes. Some specialised SHAP algorithms can do without it.
    if model.data_type == "tabular" and not model.has_reference_data:
        reasons.append("Luồng tabular này yêu cầu dữ liệu tham chiếu đại diện.")
    if model.task_type == "classification" and model.prediction_output == "labels":
        reasons.append("Luồng này cần xác suất/điểm số; không dùng nhãn lớp rời rạc.")

    if method.family == "LIME":
        if model.data_type in ("text", "image") and model.task_type == "regression":
            reasons.append("LimeTextExplainer/LimeImageExplainer tiêu chuẩn ở đây dành cho classification.")
        if model.task_type == "classification" and model.prediction_output != "probabilities":
            reasons.append("LIME classification yêu cầu xác suất lớp từ prediction wrapper.")
        adapter = {"tabular": "lime.lime_tabular.LimeTabularExplainer",
                   "text": "lime.lime_text.LimeTextExplainer",
                   "image": "lime.lime_image.LimeImageExplainer"}.get(model.data_type)
        if model.data_type == "tabular" and "categorical" in model.feature_types:
            notes.append("Mã hóa categorical và khai báo categorical_features/categorical_names.")
    elif method.family == "SHAP":
        if model.data_type == "tabular":
            adapter = "shap.Explainer(predict_fn, background, algorithm='permutation')"
            if model.model_object_available and model.model_type in ("random_forest", "gradient_boosting", "decision_tree"):
                notes.append("Có thể thử TreeExplainer nếu model cụ thể được SHAP hỗ trợ; dùng fallback nếu không.")
            elif model.model_object_available and model.model_type == "linear":
                notes.append("Có thể thử LinearExplainer sau khi kiểm tra estimator và bộ tiền xử lý.")
            if "categorical" in model.feature_types:
                notes.append("Chuẩn bị encoding/wrapper và masker phù hợp cho các biến categorical.")
        elif model.data_type in ("text", "image"):
            adapter = f"shap.Explainer + shap.maskers.{model.data_type.title()}"
            notes.append("Cần tự cấu hình tokenizer/segmentation và masker đúng miền dữ liệu.")
    elif method.family == "ALE":
        if model.data_type == "tabular" and "numerical" not in model.feature_types:
            reasons.append("ALE số cần ít nhất một biến numerical.")
        if model.data_type == "tabular" and "categorical" in model.feature_types:
            notes.append("Chỉ tính ALE cho cột numerical; giữ các cột categorical làm điều kiện đầu vào.")
        adapter = "alibi.explainers.ALE (numerical features only)"
    notes.append("Đánh giá dựa trên khai báo; cần kiểm tra prediction wrapper, shape và phiên bản thư viện khi tích hợp.")
    return CompatibilityResult(method.method_id, not reasons, tuple(reasons), tuple(notes), adapter)


def compatible_methods(context: BusinessContext, method_ids=None):
    names = normalize_methods(method_ids if method_ids is not None else [m.method_id for m in _BASE_METHODS])
    return tuple(check_compatibility(name, context) for name in names)
