"""Behaviour and integration checks for the four-module extension."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from business_context import BusinessContext, ModelProfile, collect_business_context
from demo_pipeline import build_demo
from feedback import FeedbackCollector, collect_feedback
from recommendation import recommend
from xai_methods import check_compatibility, get_method, list_methods
from xai_selector import demo_experiment


def context(**changes):
    profile = ModelProfile("test-model", "random_forest", "regression", "tabular", "numeric", ("numerical",))
    base = BusinessContext("Example company", "Manufacturing", "Explain demand", profile, ("Planners",))
    return replace(base, **changes)


def completed_collector(ctx=None, methods=("LIME", "SHAP", "ALE")):
    ctx = context() if ctx is None else ctx
    collector = FeedbackCollector(methods, context_id=ctx.context_id, data_kind="synthetic_test")
    collector.register_participant("R1", [3] * 8, expertise_weight=1, trust_weight=0.5)
    for method in methods:
        collector.submit_feedback("R1", method, [4] * 8, [4] * 8)
    return collector


class ContextTests(unittest.TestCase):
    def test_context_round_trip_and_fingerprint(self):
        ctx = context()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/context.json"
            ctx.save(path)
            loaded = BusinessContext.load(path)
        self.assertEqual(loaded, ctx)
        self.assertEqual(loaded.context_id, ctx.context_id)
        self.assertNotEqual(replace(ctx, ai_objective="Another objective").context_id, ctx.context_id)
        self.assertNotEqual(replace(ctx, model=replace(ctx.model, version="v2")).context_id, ctx.context_id)

    def test_context_rejects_invalid_model_and_users(self):
        for data in [dict(model_type="typo"), dict(task_type="clustering"),
                     dict(feature_types=()), dict(has_reference_data="false"),
                     dict(prediction_output="labels")]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                replace(context().model, **data)
        for data in [dict(target_users=()), dict(company_name=" "), dict(required_scope="unknown")]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                replace(context(), **data)

    def test_input_normalization_and_metadata_do_not_bias_score(self):
        ctx = context(required_scope=" local ")
        self.assertEqual(ctx.required_scope, "local")
        self.assertEqual(replace(ctx.model, model_type=" random_forest ").model_type, "random_forest")
        other = replace(ctx, industry="Healthcare", user_ai_literacy="basic")
        original = demo_experiment()  # legacy, unbound feedback
        self.assertEqual(recommend(ctx, original)["ranking"], recommend(other, original)["ranking"])

    def test_context_interactive_collection(self):
        inputs = iter(["Test company", "Supply chain", "Demand", "Planners, Managers", "RF", "v1",
                       "random_forest", "regression", "tabular", "numeric", "mixed",
                       "yes", "yes", "yes", "no", "local", "basic"])
        ctx = collect_business_context(input_fn=lambda _: next(inputs), output_fn=lambda _: None)
        self.assertEqual(ctx.target_users, ("Planners", "Managers"))
        self.assertEqual(ctx.model.feature_types, ("numerical", "categorical"))


class CompatibilityTests(unittest.TestCase):
    def test_base_catalogue_and_explicit_variant_scopes(self):
        self.assertEqual([m.method_id for m in list_methods()], ["LIME", "SHAP", "ALE"])
        self.assertEqual(get_method("shap_waterfall").scopes, ("local",))
        with self.assertRaises(ValueError):
            get_method("SHAP_typo")

    def test_tree_svm_and_nn_tabular_regression(self):
        ctx = context()
        for model_type in ("random_forest", "svm", "neural_network"):
            changed = replace(ctx, model=replace(ctx.model, model_type=model_type))
            self.assertTrue(all(check_compatibility(name, changed).eligible for name in ("LIME", "SHAP", "ALE")))

    def test_scope_local_global_and_both(self):
        self.assertFalse(check_compatibility("ALE", context(required_scope="local")).eligible)
        self.assertFalse(check_compatibility("LIME", context(required_scope="global")).eligible)
        self.assertFalse(check_compatibility("SHAP_Waterfall", context(required_scope="global")).eligible)
        for method, expected in (("LIME", False), ("SHAP", True), ("ALE", False)):
            self.assertEqual(check_compatibility(method, context(required_scope="both")).eligible, expected)

    def test_ale_mixed_data_only_explains_numeric_columns(self):
        ctx = context()
        categorical = replace(ctx, model=replace(ctx.model, feature_types=("categorical",)))
        self.assertFalse(check_compatibility("ALE", categorical).eligible)
        mixed = replace(ctx, model=replace(ctx.model, feature_types=("numerical", "categorical")))
        result = check_compatibility("ALE", mixed)
        self.assertTrue(result.eligible)
        self.assertTrue(any("Chỉ tính ALE" in note for note in result.notes))

    def test_lime_needs_probabilities_not_only_scores_or_labels(self):
        ctx = context()
        for output, expected in (("probabilities", True), ("scores", False), ("labels", False)):
            model = replace(ctx.model, task_type="classification", prediction_output=output)
            self.assertEqual(check_compatibility("LIME", replace(ctx, model=model)).eligible, expected)

    def test_domain_adapters_and_unsupported_time_series(self):
        ctx = context()
        for domain in ("text", "image"):
            model = replace(ctx.model, data_type=domain, feature_types=(), task_type="classification",
                            prediction_output="probabilities", has_reference_data=False)
            changed = replace(ctx, model=model)
            self.assertTrue(check_compatibility("LIME", changed).eligible)
            self.assertTrue(check_compatibility("SHAP", changed).eligible)
            self.assertFalse(check_compatibility("ALE", changed).eligible)
            regression = replace(changed, model=replace(model, task_type="regression", prediction_output="numeric"))
            self.assertFalse(check_compatibility("LIME", regression).eligible)
        model = replace(ctx.model, data_type="time_series", feature_types=())
        self.assertFalse(any(check_compatibility(name, replace(ctx, model=model)).eligible
                             for name in ("LIME", "SHAP", "ALE")))

    def test_missing_predictor_reference_or_preprocessing(self):
        for field in ("has_prediction_function", "has_reference_data", "preprocessing_ready"):
            ctx = context()
            ctx = replace(ctx, model=replace(ctx.model, **{field: False}))
            self.assertFalse(any(check_compatibility(name, ctx).eligible for name in ("LIME", "SHAP", "ALE")))


class FeedbackTests(unittest.TestCase):
    def test_staged_collection_and_detached_exports(self):
        collector = FeedbackCollector(["LIME", "SHAP"])
        with self.assertRaisesRegex(ValueError, "Trust Before"):
            collector.submit_feedback("R1", "LIME", [4] * 8, [4] * 8)
        collector.register_participant("R1", [3] * 8)
        collector.submit_feedback("R1", "LIME", [4] * 8, [4] * 8)
        self.assertEqual(collector.missing_feedback(), {"R1": ["SHAP"]})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            collector.to_experiment()
        collector.submit_feedback("R1", "SHAP", [5] * 8, [5] * 8)
        exported = collector.to_experiment()
        exported["participants"][0]["trust_before"][0] = 1
        self.assertEqual(collector.to_experiment()["participants"][0]["trust_before"][0], 3)
        self.assertEqual(collector.calculate_dots()["best_methods"], ["SHAP"])

    def test_duplicate_baseline_and_after_feedback_are_rejected(self):
        collector = completed_collector(methods=("LIME",))
        with self.assertRaises(ValueError):
            collector.register_participant("R1", [1] * 8)
        with self.assertRaises(ValueError):
            collector.submit_feedback("R1", "LIME", [5] * 8, [5] * 8)

    def test_invalid_submission_does_not_partially_change_state(self):
        collector = FeedbackCollector(["LIME"])
        collector.register_participant("R1", [3] * 8)
        with self.assertRaises(ValueError):
            collector.submit_feedback("R1", "LIME", [4] * 8, [6] * 8)
        self.assertEqual(collector.missing_feedback(), {"R1": ["LIME"]})

    def test_incomplete_snapshot_can_be_resumed(self):
        collector = FeedbackCollector(["LIME", "SHAP"])
        collector.register_participant("R1", [3] * 8)
        collector.submit_feedback("R1", "LIME", [4] * 8, [4] * 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            collector.save(path, allow_incomplete=True)
            resumed = FeedbackCollector.load(path)
        self.assertFalse(resumed.is_complete)
        resumed.submit_feedback("R1", "SHAP", [4] * 8, [4] * 8)
        self.assertTrue(resumed.is_complete)

    def test_legacy_import_and_custom_scales(self):
        raw = demo_experiment()
        imported = FeedbackCollector.from_experiment(raw)
        self.assertEqual(imported.calculate_dots()["best_methods"], ["ALE"])
        collector = FeedbackCollector(["ALE"], trust_scale={"min": 0, "max": 10, "items": 2},
                                      satisfaction_scale={"min": 1, "max": 7, "items": 3})
        collector.register_participant("R1", [0, 0])
        collector.submit_feedback("R1", "ALE", [10, 10], [7, 7, 7])
        self.assertEqual(collector.calculate_dots()["ranking"][0]["dots"], 1)

    def test_wrong_pool_and_duplicate_aliases(self):
        with self.assertRaises(ValueError):
            FeedbackCollector(["SHAP", "shap"])
        collector = FeedbackCollector(["LIME"])
        collector.register_participant("R1", [3] * 8)
        with self.assertRaises(ValueError):
            collector.submit_feedback("R1", "ALE", [4] * 8, [4] * 8)

    def test_interactive_collection_filters_before_asking(self):
        inputs = iter(["R1", "1", "0.5"] + ["3 3 3 3 3 3 3 3"] * 5)
        result = collect_feedback(context(required_scope="local"), participant_count=1,
                                  input_fn=lambda _: next(inputs), output_fn=lambda _: None)
        self.assertEqual(result.method_ids, ("LIME", "SHAP"))
        self.assertTrue(result.is_complete)


class RecommendationTests(unittest.TestCase):
    def test_highest_dots_cannot_override_technical_failure(self):
        ctx, survey, report = build_demo()
        original = deepcopy(survey)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["best_methods"], ["LIME"])
        self.assertEqual(report["dots_report"]["best_methods"], ["ALE"])
        self.assertEqual(report["excluded_methods"][0]["method_id"], "ALE")
        self.assertEqual(recommend(ctx, survey), report)
        self.assertEqual(survey, original)

    def test_no_survey_means_feedback_required_not_zero_score(self):
        report = recommend(context())
        self.assertEqual(report["status"], "feedback_required")
        self.assertEqual(report["ranking"], [])
        self.assertEqual(report["best_methods"], [])
        self.assertEqual(report["awaiting_feedback"], ["LIME", "SHAP", "ALE"])

    def test_weight_override_requires_feedback_and_preserves_original_dots(self):
        ctx = context()
        with self.assertRaisesRegex(ValueError, "requires feedback"):
            recommend(ctx, trust_weight=0.8)
        data = demo_experiment()
        result = recommend(ctx, data, trust_weight=1)
        self.assertEqual(result["best_methods"], ["SHAP_Waterfall"])
        self.assertAlmostEqual(result["ranking"][0]["dots"], 0.75)

    def test_unassessed_candidate_is_reported_as_partial(self):
        ctx = context()
        experiment = completed_collector(ctx, ("LIME",)).to_experiment()
        report = recommend(ctx, experiment, ["LIME", "SHAP", "ALE"])
        self.assertEqual(report["status"], "partial_feedback")
        self.assertFalse(report["recommendation_complete"])
        self.assertEqual(report["awaiting_feedback"], ["SHAP", "ALE"])

    def test_no_compatible_methods_and_ties(self):
        ctx = context()
        data = completed_collector(ctx).to_experiment()
        report = recommend(ctx, data)
        self.assertEqual(report["best_methods"], ["ALE", "LIME", "SHAP"])
        blocked = replace(ctx, model=replace(ctx.model, has_prediction_function=False))
        self.assertEqual(recommend(blocked)["status"], "no_compatible_method")

    def test_context_mismatch_and_unknown_methods_raise(self):
        ctx = context()
        data = completed_collector(ctx).to_experiment()
        with self.assertRaisesRegex(ValueError, "context_id"):
            recommend(replace(ctx, industry="Another industry"), data)
        with self.assertRaises(ValueError):
            recommend(ctx, data, method_ids=["LIME"])
        data["participants"][0]["responses"]["TY PO"] = {"trust_after": [4] * 8, "satisfaction": [4] * 8}
        with self.assertRaisesRegex(ValueError, "Unknown"):
            recommend(ctx, data)

    def test_incomplete_cohort_is_never_silently_dropped(self):
        data = demo_experiment()
        del data["participants"][1]["responses"]["LIME"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            recommend(context(), data)

    def test_import_keeps_shap_variants_distinct(self):
        collector = completed_collector(methods=("SHAP_Waterfall", "SHAP_Summary"))
        report = recommend(context(), collector.to_experiment())
        self.assertEqual(set(report["best_methods"]), {"SHAP_Waterfall", "SHAP_Summary"})


if __name__ == "__main__":
    unittest.main()
