"""Combine technical eligibility with original DoTS, without changing the metric.

Policy: technical filtering -> DoTS ranking of compatible, evaluated candidates.
This policy is an engineering extension, not an additional formula from the paper.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from business_context import BusinessContext
from feedback import FeedbackCollector
from xai_methods import compatible_methods, list_methods, normalize_methods
from xai_selector import evaluate


def recommend(context: BusinessContext, experiment=None, method_ids=None, trust_weight=None):
    """Return reasons, scores, ties, and unevaluated candidates explicitly.

    With an experiment, its declared pool (or observed method names for legacy
    JSON) is the default candidate pool. Without one, default = LIME, SHAP, ALE.
    Pass method_ids explicitly to include additional unassessed candidates.
    """
    if not isinstance(context, BusinessContext):
        raise ValueError("context: expected a BusinessContext")
    if trust_weight is not None and experiment is None:
        raise ValueError("trust_weight requires feedback")
    dots_report = None
    declared_pool = None
    notes = ["Technical compatibility is based on declared metadata, not a runtime explainer test."]
    if experiment is not None:
        # Normalizes labels while preserving SHAP visualization variants separately.
        collector = FeedbackCollector.from_experiment(experiment)
        if collector.context_id is not None and collector.context_id != context.context_id:
            raise ValueError("Feedback context_id does not match the business/model context")
        if collector.context_id is None:
            notes.append("Legacy feedback has no context_id; verify it belongs to this model and business context.")
        normalized = collector.to_experiment()  # Never silently drop incomplete people or methods.
        declared_pool = collector.method_ids
        dots_report = evaluate(normalized, trust_weight=trust_weight)
    pool = normalize_methods(method_ids if method_ids is not None else
                             (declared_pool if declared_pool is not None else [m.method_id for m in list_methods()]))
    scored = {} if dots_report is None else {row["method"]: row for row in dots_report["ranking"]}
    unexpected = set(scored) - set(pool)
    if unexpected:
        raise ValueError(f"Feedback contains methods outside the candidate pool: {sorted(unexpected)}")
    checks = compatible_methods(context, pool)
    ranking, excluded, awaiting = [], [], []
    for check in checks:
        if not check.eligible:
            excluded.append({**check.to_dict(), "dots": scored.get(check.method_id, {}).get("dots")})
        elif check.method_id not in scored:
            awaiting.append(check.method_id)
        else:
            row = dict(scored[check.method_id])
            row.update({"recommendation_score": row["dots"], "eligible": True,
                        "suggested_adapter": check.suggested_adapter, "notes": list(check.notes)})
            ranking.append(row)
    ranking.sort(key=lambda row: (-row["dots"], row["method"]))
    best_methods = []
    if ranking:
        best = ranking[0]["dots"]
        best_methods = [row["method"] for row in ranking
                        if math.isclose(row["dots"], best, rel_tol=0, abs_tol=1e-12)]
    if not any(check.eligible for check in checks):
        status = "no_compatible_method"
    elif not ranking:
        status = "feedback_required"
    elif awaiting:
        status = "partial_feedback"
    else:
        status = "ready"
    return {
        "status": status,
        "scoring_policy": "technical_eligibility_then_original_dots",
        "recommendation_complete": status == "ready",
        "context_id": context.context_id,
        "context": context.to_dict(),
        "data_kind": "no_feedback" if dots_report is None else dots_report["data_kind"],
        "candidate_methods": list(pool),
        "best_methods": best_methods,
        "ranking": ranking,
        "awaiting_feedback": awaiting,
        "excluded_methods": excluded,
        "compatibility": [check.to_dict() for check in checks],
        "dots_report": dots_report,
        "notes": notes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recommend technically compatible XAI methods using DoTS.")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--trust-weight", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        context = BusinessContext.load(args.context)
        experiment = None if args.feedback is None else json.loads(args.feedback.read_text(encoding="utf-8-sig"))
        if args.trust_weight is not None and experiment is None:
            raise ValueError("--trust-weight requires --feedback")
        result = recommend(context, experiment, args.methods, args.trust_weight)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Status: {result['status']}")
    for row in result["ranking"]:
        print(f"{row['method']}: DoTS={row['dots']:.6f}")
    print("Best among evaluated compatible methods: " + (", ".join(result["best_methods"]) or "none"))
    if result["awaiting_feedback"]:
        print("Awaiting feedback: " + ", ".join(result["awaiting_feedback"]))
    for row in result["excluded_methods"]:
        print(f"Excluded {row['method_id']}: " + "; ".join(row["reasons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
