"""DoTS-based XAI selection, independently implemented from Reis et al. (2025).

Reference: https://doi.org/10.1016/j.compind.2024.104233
Equations (1), (2), (4) and Algorithm 1. Python 3.10+, no dependencies.
This module ranks existing explanations using human survey responses; it does
not generate explanations or reproduce the authors' business experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number")
    return result


def _weight(value: Any, label: str, *, positive: bool = False) -> float:
    result = _number(value, label)
    if not (0 < result <= 1 if positive else 0 <= result <= 1):
        bounds = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{label}: must be in {bounds}")
    return result


def _scale(config: Any, label: str) -> tuple[float, float, int]:
    if not isinstance(config, dict):
        raise ValueError(f"{label}: expected an object")
    low = _number(config.get("min", 1), f"{label}.min")
    high = _number(config.get("max", 5), f"{label}.max")
    count = config.get("items", 8)
    span = high - low
    if not math.isfinite(span) or span <= 0 or not math.isfinite(2 * span):
        raise ValueError(f"{label}: max must exceed min with a finite span")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{label}.items: expected a positive integer")
    return low, high, count


def _ratings(values: Any, scale: tuple[float, float, int], label: str) -> list[float]:
    low, high, count = scale
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{label}: expected exactly {count} answers")
    answers = [_number(value, f"{label}[{i}]") for i, value in enumerate(values)]
    if any(value < low or value > high for value in answers):
        raise ValueError(f"{label}: all answers must be in [{low}, {high}]")
    return answers


def evaluate(experiment: dict, trust_weight: float | None = None) -> dict:
    """Return a ranking, all tied best methods, and per-respondent scores.

    Required input structure is demonstrated by demo_experiment(). The same
    respondents must rate every method using the same question order/scales.
    Ratings must already be scored so that larger values mean higher trust or
    satisfaction. No automatic reversal/imputation is performed.

    trust_weight overrides w_I for every respondent (sensitivity analysis);
    otherwise each respondent's trust_weight is used. expertise_weight is w_E.
    """
    if not isinstance(experiment, dict):
        raise ValueError("experiment: expected an object")
    trust_scale = _scale(experiment.get("trust_scale", {}), "trust_scale")
    satisfaction_scale = _scale(
        experiment.get("satisfaction_scale", {}), "satisfaction_scale"
    )
    override = None
    if trust_weight is not None:
        override = _weight(trust_weight, "trust_weight override")
    participants = experiment.get("participants")
    if not isinstance(participants, list) or not participants:
        raise ValueError("participants: expected a nonempty list")

    expected_methods: set[str] | None = None
    identifiers: set[str] = set()
    details: dict[str, list[dict]] = {}
    trust_span = trust_scale[1] - trust_scale[0]
    satisfaction_span = satisfaction_scale[1] - satisfaction_scale[0]

    for index, participant in enumerate(participants):
        label = f"participants[{index}]"
        if not isinstance(participant, dict):
            raise ValueError(f"{label}: expected an object")
        identifier = participant.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{label}.id: expected a nonempty string")
        if identifier in identifiers:
            raise ValueError(f"duplicate participant id: {identifier}")
        identifiers.add(identifier)
        expertise = _weight(
            participant.get("expertise_weight"), f"{label}.expertise_weight", positive=True
        )
        importance = _weight(
            participant.get("trust_weight", 0.5), f"{label}.trust_weight"
        )
        if override is not None:
            importance = override
        before = _ratings(participant.get("trust_before"), trust_scale, f"{label}.trust_before")
        responses = participant.get("responses")
        if not isinstance(responses, dict) or not responses:
            raise ValueError(f"{label}.responses: expected a nonempty object")
        if any(not isinstance(name, str) or not name.strip() for name in responses):
            raise ValueError(f"{label}.responses: method names must be nonempty strings")
        methods = set(responses)
        if expected_methods is None:
            expected_methods = methods
            details = {name: [] for name in sorted(methods)}
        elif methods != expected_methods:
            raise ValueError(
                f"{label}: every participant must rate the same methods; "
                f"missing={sorted(expected_methods - methods)}, extra={sorted(methods - expected_methods)}"
            )

        for method, response in responses.items():
            response_label = f"{label}.responses.{method}"
            if not isinstance(response, dict):
                raise ValueError(f"{response_label}: expected an object")
            after = _ratings(response.get("trust_after"), trust_scale, f"{response_label}.trust_after")
            satisfaction = _ratings(
                response.get("satisfaction"), satisfaction_scale, f"{response_label}.satisfaction"
            )
            # Eq. (1): normalize the CHANGE, not the absolute post-explanation trust.
            trust = math.fsum(
                ((post - pre) + trust_span) / (2 * trust_span)
                for pre, post in zip(before, after)
            ) / len(before)
            # Eq. (2): satisfaction uses the ordinary min-max normalization.
            sat = math.fsum(
                (value - satisfaction_scale[0]) / satisfaction_span
                for value in satisfaction
            ) / len(satisfaction)
            # Eq. (4): first mix T and S for this particular respondent.
            combined = importance * trust + (1 - importance) * sat
            details[method].append({
                "participant_id": identifier,
                "trust": trust,
                "satisfaction": sat,
                "trust_change_raw": math.fsum(
                    (post - pre) / len(before) for pre, post in zip(before, after)
                ),
                "expertise_weight": expertise,
                "trust_weight": importance,
                "combined_score": combined,
            })

    ranking = []
    for method, rows in details.items():
        total_weight = math.fsum(row["expertise_weight"] for row in rows)
        # Normalize weights first to avoid underflow with tiny positive weights.
        weighted = lambda key: math.fsum(
            (row["expertise_weight"] / total_weight) * row[key] for row in rows
        )
        ranking.append({
            "method": method,
            "dots": weighted("combined_score"),
            # These two weighted summaries are descriptive, not separate Eq. (4) inputs.
            "trust_weighted": weighted("trust"),
            "satisfaction_weighted": weighted("satisfaction"),
            "respondents": len(rows),
        })
    ranking.sort(key=lambda row: (-row["dots"], row["method"]))
    best = ranking[0]["dots"]
    return {
        "data_kind": experiment.get("data_kind", "unspecified"),
        "best_methods": [row["method"] for row in ranking
                         if math.isclose(row["dots"], best, rel_tol=0, abs_tol=1e-12)],
        "ranking": ranking,
        "per_respondent": details,
    }


def sensitivity(experiment: dict, weights=None) -> list[dict]:
    """Evaluate a common w_I grid; does not modify the input experiment."""
    if weights is None:
        weights = [i / 40 for i in range(41)]
    result = []
    for value in weights:
        report = evaluate(experiment, trust_weight=value)
        result.append({"trust_weight": float(value), "best_methods": report["best_methods"],
                       "scores": {row["method"]: row["dots"] for row in report["ranking"]}})
    return result


def demo_experiment() -> dict:
    """Artificial survey scores only; no observations from the paper."""
    methods = ["LIME", "SHAP_Waterfall", "ALE"]
    # Each method has a post-trust vector and a satisfaction vector.
    sample = [
        ("R1", 1.0, [3, 3, 3, 3, 3, 3, 3, 3], [
            ([4, 4, 4, 5, 4, 4, 4, 3], [4, 4, 4, 4, 5, 4, 4, 3]),
            ([5, 5, 4, 5, 5, 4, 5, 5], [3, 3, 4, 3, 3, 4, 3, 3]),
            ([4, 4, 4, 4, 4, 3, 4, 4], [5, 5, 5, 5, 5, 5, 5, 5]),
        ]),
        ("R2", 0.75, [2, 3, 2, 3, 2, 3, 2, 3], [
            ([4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4]),
            ([5, 5, 5, 5, 5, 5, 5, 5], [3, 3, 3, 4, 3, 3, 3, 4]),
            ([3, 4, 3, 4, 3, 4, 3, 4], [5, 5, 5, 5, 5, 5, 5, 4]),
        ]),
        ("R3", 0.25, [3, 3, 3, 3, 3, 3, 3, 3], [
            ([4, 4, 4, 4, 4, 4, 4, 4], [5, 4, 5, 4, 5, 4, 5, 4]),
            ([4, 5, 4, 5, 4, 5, 4, 5], [2, 3, 2, 3, 2, 3, 2, 3]),
            ([3, 4, 3, 4, 3, 4, 3, 4], [4, 5, 4, 5, 4, 5, 4, 5]),
        ]),
    ]
    participants = []
    for identifier, expertise, before, answers in sample:
        participants.append({
            "id": identifier, "expertise_weight": expertise, "trust_weight": 0.5,
            "trust_before": before,
            "responses": {
                method: {"trust_after": post, "satisfaction": sat}
                for method, (post, sat) in zip(methods, answers)
            },
        })
    return {"data_kind": "synthetic_demo",
            "trust_scale": {"min": 1, "max": 5, "items": 8},
            "satisfaction_scale": {"min": 1, "max": 5, "items": 8},
            "participants": participants}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rank XAI methods using human-feedback DoTS.")
    parser.add_argument("--input", type=Path, help="Survey JSON; omitted = synthetic demo")
    parser.add_argument("--trust-weight", type=float, help="Override w_I for all respondents")
    parser.add_argument("--output", type=Path, help="Write the complete JSON report")
    parser.add_argument("--sensitivity", type=Path, help="Write scores for w_I=0..1, step 0.025")
    parser.add_argument("--write-example", type=Path, help="Save a synthetic input JSON")
    args = parser.parse_args(argv)
    try:
        if args.input is None:
            experiment = demo_experiment()
        else:
            experiment = json.loads(args.input.read_text(encoding="utf-8-sig"))
        report = evaluate(experiment, args.trust_weight)
        if args.output:
            _write_json(args.output, report)
        if args.sensitivity:
            _write_json(args.sensitivity, sensitivity(experiment))
        if args.write_example:
            _write_json(args.write_example, demo_experiment())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.input is None:
        print("SYNTHETIC DEMO -- not results from the paper")
    print(f"{'Method':<24} {'DoTS':>9} {'Trust(wE)':>12} {'Satisf.(wE)':>12}")
    for row in report["ranking"]:
        print(f"{row['method']:<24} {row['dots']:9.6f} "
              f"{row['trust_weighted']:12.6f} {row['satisfaction_weighted']:12.6f}")
    print("Selected: " + ", ".join(report["best_methods"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
