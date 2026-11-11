"""Mathematical regression and validation tests; no external dependencies."""

from copy import deepcopy
import math
import unittest

from xai_selector import demo_experiment, evaluate, sensitivity


def one_person(before=3, after=3, satisfaction=3, importance=0.5, expertise=1.0):
    return {"participants": [{
        "id": "R1", "expertise_weight": expertise, "trust_weight": importance,
        "trust_before": [before] * 8,
        "responses": {"A": {"trust_after": [after] * 8,
                             "satisfaction": [satisfaction] * 8}},
    }]}


def score(data, **kwargs):
    return evaluate(data, **kwargs)["ranking"][0]["dots"]


class DoTSTests(unittest.TestCase):
    def test_unchanged_trust_is_half_not_zero(self):
        result = evaluate(one_person(importance=1))
        self.assertEqual(result["per_respondent"]["A"][0]["trust"], 0.5)
        self.assertEqual(score(one_person(importance=1)), 0.5)

    def test_extremes(self):
        self.assertEqual(score(one_person(before=1, after=5, satisfaction=5)), 1)
        self.assertEqual(score(one_person(before=5, after=1, satisfaction=1)), 0)

    def test_manual_calculation(self):
        # T=((4-2)+4)/8=.75, S=(4-1)/4=.75; DoTS=.75.
        self.assertEqual(score(one_person(before=2, after=4, satisfaction=4)), 0.75)

    def test_nonuniform_questions(self):
        data = one_person()
        person = data["participants"][0]
        person["trust_before"] = [1, 2, 3, 4, 5, 1, 2, 3]
        person["responses"]["A"]["trust_after"] = [2, 2, 4, 4, 5, 3, 4, 4]
        person["responses"]["A"]["satisfaction"] = [1, 2, 3, 4, 5, 4, 3, 2]
        # Sum of trust changes=7 -> T=39/64; satisfaction sum=24 -> S=.5.
        self.assertAlmostEqual(score(data), (39 / 64 + 0.5) / 2)

    def test_paper_expertise_example(self):
        # Section 3.3.3 Example 1: two maximal scores and one minimal score.
        good = one_person(before=1, after=5, satisfaction=5, expertise=0.25)["participants"][0]
        good2 = deepcopy(good)
        good2["id"] = "R2"
        bad = one_person(before=5, after=1, satisfaction=1)["participants"][0]
        bad["id"] = "R3"
        data = {"participants": [good, good2, bad]}
        self.assertAlmostEqual(score(data), 1 / 3)
        good["expertise_weight"] = good2["expertise_weight"] = 1
        bad["expertise_weight"] = 0.25
        self.assertAlmostEqual(score(data), 8 / 9)

    def test_individual_weights_mixed_before_aggregation(self):
        # R1 uses trust only (1); R2 uses satisfaction only (1).
        # Averaging T and S first would incorrectly return .5.
        a = one_person(before=1, after=5, satisfaction=1, importance=1)["participants"][0]
        b = one_person(before=5, after=1, satisfaction=5, importance=0)["participants"][0]
        b["id"] = "R2"
        self.assertEqual(score({"participants": [a, b]}), 1)

    def test_custom_scales(self):
        data = one_person()
        data["trust_scale"] = {"min": 0, "max": 10, "items": 2}
        data["satisfaction_scale"] = {"min": 1, "max": 7, "items": 3}
        p = data["participants"][0]
        p["trust_before"] = [0, 0]
        p["responses"]["A"] = {"trust_after": [10, 10], "satisfaction": [7, 7, 7]}
        self.assertEqual(score(data), 1)

    def test_ties_are_reported(self):
        data = one_person()
        data["participants"][0]["responses"]["B"] = deepcopy(
            data["participants"][0]["responses"]["A"]
        )
        self.assertEqual(evaluate(data)["best_methods"], ["A", "B"])

    def test_demo_hand_calculated_scores_and_sensitivity(self):
        data = demo_experiment()
        original = deepcopy(data)
        result = evaluate(data)
        scores = {row["method"]: row["dots"] for row in result["ranking"]}
        self.assertAlmostEqual(scores["ALE"], 0.791015625)
        self.assertAlmostEqual(scores["LIME"], 0.70703125)
        self.assertAlmostEqual(scores["SHAP_Waterfall"], 0.64453125)
        grid = sensitivity(data)
        self.assertEqual(len(grid), 41)
        self.assertEqual(grid[0]["best_methods"], ["ALE"])
        self.assertEqual(grid[-1]["best_methods"], ["SHAP_Waterfall"])
        for row in grid:
            for value in row["scores"].values():
                self.assertTrue(0 <= value <= 1)
        self.assertEqual(data, original)

    def test_reject_missing_methods_and_duplicate_respondents(self):
        data = demo_experiment()
        del data["participants"][1]["responses"]["ALE"]
        with self.assertRaisesRegex(ValueError, "same methods"):
            evaluate(data)
        data = demo_experiment()
        data["participants"][1]["id"] = "R1"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(data)

    def test_reject_invalid_ratings(self):
        for bad in [None, [], [1] * 7, [6] * 8, [0] * 8,
                    [math.nan] * 8, [math.inf] * 8, [True] * 8, ["3"] * 8]:
            with self.subTest(bad=bad):
                data = one_person()
                data["participants"][0]["trust_before"] = bad
                with self.assertRaises(ValueError):
                    evaluate(data)

    def test_reject_invalid_weights(self):
        for field, values in {
            "expertise_weight": [0, -1, 2, math.nan, True, "1"],
            "trust_weight": [-1, 2, math.inf, True, "0.5"],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data = one_person()
                    data["participants"][0][field] = value
                    with self.assertRaises(ValueError):
                        evaluate(data)
        with self.assertRaises(ValueError):
            evaluate(one_person(), trust_weight=1.1)

    def test_reject_bad_scale_and_empty_input(self):
        for scale in [{"min": 5, "max": 5}, {"items": 0}, {"items": True},
                      {"min": 5, "max": 1}, {"min": -1e308, "max": 1e308}]:
            data = one_person()
            data["trust_scale"] = scale
            with self.assertRaises(ValueError):
                evaluate(data)
        for data in [{}, {"participants": []}, [], {"participants": [None]}]:
            with self.assertRaises(ValueError):
                evaluate(data)

    def test_participant_order_does_not_change_scores(self):
        data = demo_experiment()
        expected = evaluate(data)["ranking"]
        data["participants"].reverse()
        self.assertEqual(evaluate(data)["ranking"], expected)


if __name__ == "__main__":
    unittest.main()
