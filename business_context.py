"""Business and declared model context; Python 3.10+, standard library only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


MODEL_TYPES = ("random_forest", "gradient_boosting", "decision_tree", "linear",
               "svm", "knn", "neural_network", "other")
DATA_TYPES = ("tabular", "text", "image", "time_series")
TASK_TYPES = ("classification", "regression")
SCOPES = ("any", "local", "global", "both")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a nonempty string")
    return value.strip()


def _choice(value, choices, label):
    value = _text(value, label)
    if value not in choices:
        raise ValueError(f"{label}: choose one of {', '.join(choices)}")
    return value


def _strings(value, label, *, nonempty=True):
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f"{label}: expected a {'nonempty ' if nonempty else ''}list")
    result = tuple(_text(item, label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label}: duplicate values")
    return result


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_type: str
    task_type: str
    data_type: str
    prediction_output: str
    feature_types: tuple[str, ...] = ()
    has_prediction_function: bool = True
    has_reference_data: bool = True
    preprocessing_ready: bool = True
    model_object_available: bool = False
    version: str = "unspecified"

    def __post_init__(self):
        object.__setattr__(self, "name", _text(self.name, "model.name"))
        object.__setattr__(self, "version", _text(self.version, "model.version"))
        for field, choices in (("model_type", MODEL_TYPES), ("task_type", TASK_TYPES), ("data_type", DATA_TYPES)):
            object.__setattr__(self, field, _choice(getattr(self, field), choices, f"model.{field}"))
        allowed_outputs = ("probabilities", "scores", "labels") if self.task_type == "classification" else ("numeric",)
        object.__setattr__(self, "prediction_output", _choice(self.prediction_output, allowed_outputs, "model.prediction_output"))
        features = _strings(self.feature_types, "model.feature_types", nonempty=self.data_type == "tabular")
        if any(item not in ("numerical", "categorical") for item in features):
            raise ValueError("model.feature_types: use numerical and/or categorical")
        if self.data_type != "tabular" and features:
            raise ValueError("model.feature_types: leave empty for non-tabular data")
        object.__setattr__(self, "feature_types", features)
        for field in ("has_prediction_function", "has_reference_data", "preprocessing_ready", "model_object_available"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"model.{field}: expected true/false")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("model: expected an object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"model: missing or unknown fields: {exc}") from exc


@dataclass(frozen=True)
class BusinessContext:
    company_name: str
    industry: str
    ai_objective: str
    model: ModelProfile
    target_users: tuple[str, ...]
    required_scope: str = "any"
    user_ai_literacy: str = "intermediate"

    def __post_init__(self):
        for field in ("company_name", "industry", "ai_objective"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if not isinstance(self.model, ModelProfile):
            raise ValueError("model: expected a ModelProfile")
        object.__setattr__(self, "target_users", _strings(self.target_users, "target_users"))
        object.__setattr__(self, "required_scope", _choice(self.required_scope, SCOPES, "required_scope"))
        object.__setattr__(self, "user_ai_literacy", _choice(self.user_ai_literacy, ("basic", "intermediate", "advanced"), "user_ai_literacy"))

    def to_dict(self):
        # Return JSON-native lists, with no references to mutable internal data.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))

    @property
    def context_id(self):
        """Fingerprint prevents accidentally reusing feedback for another context."""
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("context: expected an object")
        values = dict(data)
        values["model"] = ModelProfile.from_dict(values.get("model"))
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"context: missing or unknown fields: {exc}") from exc

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _prompt(prompt, validator, input_fn, output_fn):
    while True:
        try:
            return validator(input_fn(prompt))
        except ValueError as exc:
            output_fn(f"Dữ liệu chưa hợp lệ: {exc}")


def collect_business_context(input_fn=input, output_fn=print):
    """Collect context in a terminal; input/output functions can be replaced by a UI."""
    def text(label):
        return _prompt(label + ": ", lambda value: _text(value, label), input_fn, output_fn)

    def choice(label, options):
        return _prompt(f"{label} ({'/'.join(options)}): ",
                       lambda value: _choice(value, options, label), input_fn, output_fn)

    def yes_no(label):
        return choice(label, ("yes", "no")) == "yes"

    company = text("Tên doanh nghiệp")
    industry = text("Lĩnh vực")
    objective = text("Mục tiêu ứng dụng AI")
    users = _prompt("Đối tượng sử dụng (ngăn cách bằng dấu phẩy): ",
                    lambda value: _strings(value.split(","), "target_users"), input_fn, output_fn)
    name = text("Tên mô hình")
    version = text("Phiên bản mô hình")
    model_type = choice("Loại mô hình", MODEL_TYPES)
    task = choice("Bài toán", TASK_TYPES)
    data_type = choice("Loại dữ liệu đầu vào của bộ giải thích", DATA_TYPES)
    outputs = ("probabilities", "scores", "labels") if task == "classification" else ("numeric",)
    output = choice("Đầu ra mô hình", outputs)
    features = ()
    if data_type == "tabular":
        feature_choice = choice("Các loại biến", ("numerical", "categorical", "mixed"))
        features = ("numerical", "categorical") if feature_choice == "mixed" else (feature_choice,)
    model = ModelProfile(
        name=name, version=version, model_type=model_type, task_type=task,
        data_type=data_type, prediction_output=output, feature_types=features,
        has_prediction_function=yes_no("Có hàm dự đoán cho nhiều mẫu?"),
        has_reference_data=yes_no("Có dữ liệu tham chiếu đại diện?"),
        preprocessing_ready=yes_no("Bộ tiền xử lý nhận được đầu vào đã biến đổi/che bớt?"),
        model_object_available=yes_no("Có truy cập trực tiếp đối tượng mô hình?"),
    )
    return BusinessContext(company, industry, objective, model, users,
                           choice("Phạm vi giải thích cần dùng", SCOPES),
                           choice("Hiểu biết AI của người dùng", ("basic", "intermediate", "advanced")))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect a business/model context.")
    parser.add_argument("--output", type=Path, default=Path("business_context.json"))
    args = parser.parse_args(argv)
    try:
        context = collect_business_context()
        context.save(args.output)
    except (OSError, ValueError, EOFError) as exc:
        parser.error(str(exc))
    print(f"Đã lưu bối cảnh: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
