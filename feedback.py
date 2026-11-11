"""Staged survey collection, JSON persistence, and DoTS input conversion.

Questionnaire texts are intentionally not invented here. Enter already-scored
answers in the validated instrument's item order, with higher = more positive.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from business_context import BusinessContext, _prompt, _text
from xai_methods import compatible_methods, get_method, normalize_methods
from xai_selector import _ratings, _scale, _weight, evaluate


class FeedbackCollector:
    """Freeze each baseline before collecting after-trust and satisfaction.

    Each participant must rate the whole selected pool before DoTS export.
    IDs are pseudonyms, not required names or email addresses.
    """

    def __init__(self, method_ids, *, context_id=None, trust_scale=None,
                 satisfaction_scale=None, data_kind="real_survey"):
        self.method_ids = normalize_methods(method_ids)
        self.context_id = None if context_id is None else _text(context_id, "context_id")
        self.data_kind = _text(data_kind, "data_kind")
        trust = _scale({} if trust_scale is None else trust_scale, "trust_scale")
        sat = _scale({} if satisfaction_scale is None else satisfaction_scale, "satisfaction_scale")
        self._trust_scale, self._sat_scale = trust, sat
        self._participants = {}

    def register_participant(self, participant_id, trust_before, *, expertise_weight=1.0, trust_weight=0.5):
        identifier = _text(participant_id, "participant_id")
        if identifier in self._participants:
            raise ValueError(f"Participant already registered: {identifier}; baseline cannot be overwritten")
        # Validate before mutating collection state, using the shared DoTS rules.
        before = _ratings(trust_before, self._trust_scale, "trust_before")
        expertise = _weight(expertise_weight, "expertise_weight", positive=True)
        importance = _weight(trust_weight, "trust_weight")
        self._participants[identifier] = {
            "id": identifier, "expertise_weight": expertise, "trust_weight": importance,
            "trust_before": before, "responses": {},
        }

    def submit_feedback(self, participant_id, method_id, trust_after, satisfaction):
        identifier = _text(participant_id, "participant_id")
        if identifier not in self._participants:
            raise ValueError("Register Trust Before before submitting Trust After/Satisfaction")
        method = get_method(method_id).method_id
        if method not in self.method_ids:
            raise ValueError(f"Method not included in this survey: {method}")
        responses = self._participants[identifier]["responses"]
        if method in responses:
            raise ValueError(f"Feedback already submitted for {identifier}/{method}")
        after = _ratings(trust_after, self._trust_scale, "trust_after")
        sat = _ratings(satisfaction, self._sat_scale, "satisfaction")
        responses[method] = {"trust_after": after, "satisfaction": sat}

    def missing_feedback(self):
        return {identifier: [name for name in self.method_ids if name not in p["responses"]]
                for identifier, p in self._participants.items()
                if any(name not in p["responses"] for name in self.method_ids)}

    @property
    def is_complete(self):
        return bool(self._participants) and not self.missing_feedback()

    def snapshot(self):
        """Copy current state, including incomplete responses, for resuming later."""
        def config(scale):
            return {"min": scale[0], "max": scale[1], "items": scale[2]}

        result = {
            "data_kind": self.data_kind, "candidate_methods": list(self.method_ids),
            "trust_scale": config(self._trust_scale), "satisfaction_scale": config(self._sat_scale),
            "participants": deepcopy(list(self._participants.values())),
        }
        if self.context_id is not None:
            result["context_id"] = self.context_id
        return result

    def to_experiment(self):
        if not self.is_complete:
            raise ValueError(f"Survey is incomplete; no participants or missing feedback: {self.missing_feedback()}")
        experiment = self.snapshot()
        evaluate(experiment)  # Keep the cohort/scales contract identical to the existing core.
        return experiment

    def calculate_dots(self, trust_weight=None):
        return evaluate(self.to_experiment(), trust_weight=trust_weight)

    def save(self, path, *, allow_incomplete=False):
        data = self.snapshot() if allow_incomplete else self.to_experiment()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def from_experiment(cls, experiment):
        if not isinstance(experiment, dict):
            raise ValueError("experiment: expected an object")
        participants = experiment.get("participants")
        if not isinstance(participants, list):
            raise ValueError("participants: expected a list")
        observed = []
        for participant in participants:
            if not isinstance(participant, dict) or not isinstance(participant.get("responses"), dict):
                raise ValueError("participant/responses: expected objects")
            for name in participant["responses"]:
                canonical = get_method(name).method_id
                if canonical not in observed:
                    observed.append(canonical)
        pool = experiment.get("candidate_methods", observed)
        collector = cls(pool, context_id=experiment.get("context_id"),
                        trust_scale=experiment.get("trust_scale", {}),
                        satisfaction_scale=experiment.get("satisfaction_scale", {}),
                        data_kind=experiment.get("data_kind", "unspecified"))
        for participant in participants:
            collector.register_participant(
                participant.get("id"), participant.get("trust_before"),
                expertise_weight=participant.get("expertise_weight"),
                trust_weight=participant.get("trust_weight", 0.5),
            )
            for name, response in participant["responses"].items():
                if not isinstance(response, dict):
                    raise ValueError("response: expected an object")
                collector.submit_feedback(participant["id"], name,
                                          response.get("trust_after"), response.get("satisfaction"))
        return collector

    @classmethod
    def load(cls, path):
        return cls.from_experiment(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def collect_feedback(context: BusinessContext, method_ids=None, participant_count=1,
                     input_fn=input, output_fn=print):
    """Interactive CLI collector; presents score slots, not new questionnaire texts."""
    if isinstance(participant_count, bool) or not isinstance(participant_count, int) or participant_count < 1:
        raise ValueError("participant_count: expected a positive integer")
    checks = compatible_methods(context, method_ids)
    candidates = [item.method_id for item in checks if item.eligible]
    for item in checks:
        if not item.eligible:
            output_fn(f"Loại {item.method_id}: {'; '.join(item.reasons)}")
    if not candidates:
        raise ValueError("No technically compatible methods for this context")
    collector = FeedbackCollector(candidates, context_id=context.context_id)

    def answers(label, scale):
        def parse(value):
            values = [float(part) for part in value.replace(",", " ").split()]
            return _ratings(values, scale, label)
        return _prompt(f"{label}: nhập {scale[2]} điểm đã chấm ({scale[0]:g}–{scale[1]:g}), cách nhau bằng dấu cách: ",
                       parse, input_fn, output_fn)

    def identifier(value):
        value = _text(value, "participant_id")
        if value in collector._participants:
            raise ValueError("ID đã được sử dụng")
        return value

    output_fn("Dùng cùng bảng hỏi và thứ tự câu; trình bày mô hình trước khi thu Trust Before.")
    for _ in range(participant_count):
        pid = _prompt("Mã người đánh giá: ", identifier, input_fn, output_fn)
        expertise = _prompt("Trọng số chuyên môn (0 < w_E <= 1): ",
                            lambda value: _weight(float(value), "w_E", positive=True), input_fn, output_fn)
        importance = _prompt("Trọng số tin cậy (0 <= w_I <= 1): ",
                             lambda value: _weight(float(value), "w_I"), input_fn, output_fn)
        before = answers("Trust Before", collector._trust_scale)
        collector.register_participant(pid, before, expertise_weight=expertise, trust_weight=importance)
        for method in candidates:
            output_fn(f"Trình bày giải thích {method} cho người đánh giá trước khi tiếp tục.")
            after = answers(f"{method} / Trust After", collector._trust_scale)
            sat = answers(f"{method} / Satisfaction", collector._sat_scale)
            collector.submit_feedback(pid, method, after, sat)
    return collector


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect Trust Before, Trust After and Satisfaction.")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--participants", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("survey.json"))
    args = parser.parse_args(argv)
    try:
        context = BusinessContext.load(args.context)
        collection = collect_feedback(context, args.methods, args.participants)
        collection.save(args.output)
    except (OSError, ValueError, EOFError) as exc:
        parser.error(str(exc))
    print(f"Đã lưu khảo sát: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
