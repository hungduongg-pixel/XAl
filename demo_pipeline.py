"""Run all four modules on synthetic data; no XAI/ML packages required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from business_context import BusinessContext, ModelProfile
from feedback import FeedbackCollector
from recommendation import recommend
from xai_selector import demo_experiment


def build_demo():
    context = BusinessContext(
        company_name="Doanh nghiệp minh họa", industry="Chuỗi cung ứng",
        ai_objective="Giải thích từng dự báo nhu cầu để nhân viên kiểm tra đơn hàng.",
        model=ModelProfile(
            name="Demand Random Forest", version="synthetic-demo-v1",
            model_type="random_forest", task_type="regression", data_type="tabular",
            prediction_output="numeric", feature_types=("numerical", "categorical"),
            model_object_available=True,
        ),
        target_users=("Nhân viên kế hoạch", "Quản lý logistics"), required_scope="local",
    )
    raw = demo_experiment()
    pool = ["LIME", "SHAP_Waterfall", "ALE"]
    collection = FeedbackCollector(pool, context_id=context.context_id, data_kind="synthetic_demo")
    # Deliberately keep the old demo pool to demonstrate technical filtering of
    # already-collected results. In a new survey, prefilter the pool first.
    for participant in raw["participants"]:
        collection.register_participant(
            participant["id"], participant["trust_before"],
            expertise_weight=participant["expertise_weight"], trust_weight=participant["trust_weight"],
        )
        for method, response in participant["responses"].items():
            collection.submit_feedback(participant["id"], method, response["trust_after"], response["satisfaction"])
    survey = collection.to_experiment()
    return context, survey, recommend(context, survey)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic four-module workflow demo.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    context, survey, result = build_demo()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        context.save(args.output_dir / "example_business_context.json")
        for name, data in (("example_feedback.json", survey), ("example_recommendation.json", result)):
            (args.output_dir / name).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("SYNTHETIC DEMO -- not research findings or real business observations")
    print("Status:", result["status"])
    for row in result["ranking"]:
        print(f"{row['method']}: DoTS={row['dots']:.6f}")
    print("Selected:", ", ".join(result["best_methods"]))
    for row in result["excluded_methods"]:
        print(f"Excluded {row['method_id']} (DoTS={row['dots']:.6f}): " + "; ".join(row["reasons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
