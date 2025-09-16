"""
Evaluation Metrics Module
=========================
Comprehensive metrics for physics QA evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from difflib import SequenceMatcher


@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    details: Dict[str, Any] | None = None
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"MetricResult(value={self.value:.3f}, confidence={self.confidence:.3f})"


class PhysicsMetrics:
    """Comprehensive metrics for physics QA evaluation."""

    # Physical units patterns
    UNIT_PATTERNS: Dict[str, str] = {
        "length": r"\b(m|meter|km|cm|mm)\b",
        "time": r"\b(s|second|min|hour|hr)\b",
        "mass": r"\b(kg|kilogram|g|gram|mg)\b",
        "force": r"\b(N|Newton|kN)\b",
        "energy": r"\b(J|Joule|kJ|eV|cal)\b",
        "power": r"\b(W|Watt|kW|MW)\b",
        "temperature": r"\b(K|Kelvin|°C|Celsius|°F)\b",
        "charge": r"\b(C|Coulomb|e)\b",
        "voltage": r"\b(V|Volt|kV|mV)\b",
        "current": r"\b(A|Ampere|mA)\b",
    }

    # Dimensional analysis patterns (kept for possible extensions)
    DIMENSION_PATTERNS: Dict[str, str] = {
        "length": r"\[L\]",
        "mass": r"\[M\]",
        "time": r"\[T\]",
        "current": r"\[I\]",
        "temperature": r"\[Θ\]",
        "amount": r"\[N\]",
        "luminosity": r"\[J\]",
    }

    @staticmethod
    def calculate_accuracy(
        predictions: List[str],
        ground_truths: List[str],
        method: str = "exact",
    ) -> MetricResult:
        """Calculate accuracy (exact / fuzzy / semantic)."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        if not predictions:
            return MetricResult(value=0.0, details={"error": "No predictions"})

        correct = 0
        samples = []

        for pred, truth in zip(predictions, ground_truths):
            if method == "exact":
                is_correct = pred.strip().lower() == truth.strip().lower()
            elif method == "fuzzy":
                is_correct = PhysicsMetrics._fuzzy_match(pred, truth)
            elif method == "semantic":
                is_correct = PhysicsMetrics._semantic_match(pred, truth)
            else:
                raise ValueError(f"Unknown method: {method}")

            if is_correct:
                correct += 1
            samples.append({"prediction": pred[:50], "truth": truth[:50], "correct": is_correct})

        return MetricResult(
            value=correct / len(predictions),
            details={
                "correct": correct,
                "total": len(predictions),
                "method": method,
                "samples": samples[:5],
            },
        )

    @staticmethod
    def _fuzzy_match(pred: str, truth: str, threshold: float = 0.85) -> bool:
        pred_clean = re.sub(r"[^\w\s]", "", pred.lower())
        truth_clean = re.sub(r"[^\w\s]", "", truth.lower())
        return SequenceMatcher(None, pred_clean, truth_clean).ratio() >= threshold

    @staticmethod
    def _semantic_match(pred: str, truth: str) -> bool:
        pred_nums = re.findall(r"[-+]?\d*\.?\d+", pred)
        truth_nums = re.findall(r"[-+]?\d*\.?\d+", truth)
        if pred_nums and truth_nums:
            try:
                pred_val = float(pred_nums[0])
                truth_val = float(truth_nums[0])
                return abs(pred_val - truth_val) / (abs(truth_val) + 1e-10) < 0.05
            except Exception:
                pass
        return PhysicsMetrics._fuzzy_match(pred, truth, threshold=0.7)

    @staticmethod
    def calculate_unit_consistency(
        responses: List[str],
        expected_units: Optional[List[List[str]]] = None,
    ) -> MetricResult:
        """Calculate unit consistency metric."""
        total = len(responses)
        if total == 0:
            return MetricResult(value=0.0)

        correct = 0
        has_units = 0
        samples = []

        for i, response in enumerate(responses):
            units_found = PhysicsMetrics._extract_units(response)

            if units_found:
                has_units += 1
                if expected_units and i < len(expected_units):
                    expected = expected_units[i]
                    if any(u in response for u in expected):
                        correct += 1
                        samples.append({"response": response[:50], "units": units_found, "correct": True})
                else:
                    # Count presence of units as "consistent" if not provided expectations
                    correct += 1
            else:
                samples.append({"response": response[:50], "units": None, "correct": False})

        return MetricResult(
            value=correct / total,
            details={
                "with_units": has_units,
                "correct_units": correct,
                "total": total,
                "unit_rate": has_units / total if total > 0 else 0.0,
                "samples": samples[:5],
            },
        )

    @staticmethod
    def _extract_units(text: str) -> List[str]:
        units_found: List[str] = []
        for _, pattern in PhysicsMetrics.UNIT_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                units_found.extend(matches)
        return sorted(set(units_found))

    @staticmethod
    def calculate_computation_accuracy(
        predictions: List[str],
        ground_truths: List[float],
        tolerance: float = 0.01,
    ) -> MetricResult:
        """Calculate computational accuracy for numerical problems."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Mismatched lengths")

        correct = 0
        extracted = 0
        samples = []

        for pred, truth in zip(predictions, ground_truths):
            numbers = re.findall(r"[-+]?\d*\.?\d+", pred)
            if not numbers:
                samples.append({"predicted": None, "truth": truth, "error": 1.0, "correct": False})
                continue

            extracted += 1
            try:
                pred_value = float(numbers[0])
                rel_err = abs(pred_value - truth) / (abs(truth) + 1e-10)
                is_ok = rel_err <= tolerance
                if is_ok:
                    correct += 1
                samples.append({"predicted": pred_value, "truth": truth, "error": rel_err, "correct": is_ok})
            except Exception:
                samples.append({"predicted": None, "truth": truth, "error": 1.0, "correct": False})

        return MetricResult(
            value=correct / len(predictions) if predictions else 0.0,
            details={
                "correct": correct,
                "extracted": extracted,
                "total": len(predictions),
                "extraction_rate": extracted / len(predictions) if predictions else 0.0,
                "tolerance": tolerance,
                "samples": samples[:5],
            },
        )

    @staticmethod
    def calculate_dimensional_analysis_score(
        responses: List[str],
        expected_dimensions: List[str],
    ) -> MetricResult:
        """Check dimensional analysis correctness by simple bracketed pattern."""
        if len(responses) != len(expected_dimensions):
            raise ValueError("Mismatched lengths")

        correct = 0
        samples = []

        for response, expected in zip(responses, expected_dimensions):
            found_dims = re.findall(r"\[[\w\s\^\-\+\*\/]+\]", response)
            is_correct = False
            if found_dims:
                found_norm = PhysicsMetrics._normalize_dimensions(found_dims[0])
                expected_norm = PhysicsMetrics._normalize_dimensions(expected)
                is_correct = found_norm == expected_norm

            if is_correct:
                correct += 1

            samples.append(
                {"response": response[:50], "found": found_dims[0] if found_dims else None, "expected": expected,
                 "correct": is_correct}
            )

        score = correct / len(responses) if responses else 0.0
        return MetricResult(value=score, details={"correct": correct, "total": len(responses), "samples": samples[:5]})

    @staticmethod
    def _normalize_dimensions(dim_expr: str) -> str:
        normalized = re.sub(r"[\[\]\s]", "", dim_expr)
        terms = re.findall(r"[A-Z][\^\-\d]*", normalized)
        terms.sort()
        return "".join(terms)

    @staticmethod
    def calculate_response_time_score(
        response_times: List[float],
        max_acceptable_time: float = 5.0,
    ) -> MetricResult:
        """Score based on fraction under a threshold; include basic stats."""
        if not response_times:
            return MetricResult(value=0.0)

        under = sum(1 for t in response_times if t <= max_acceptable_time)
        return MetricResult(
            value=under / len(response_times),
            details={
                "avg_time": float(np.mean(response_times)),
                "median_time": float(np.median(response_times)),
                "min_time": float(np.min(response_times)),
                "max_time": float(np.max(response_times)),
                "under_threshold": under,
                "total": len(response_times),
                "threshold": max_acceptable_time,
            },
        )

    @staticmethod
    def calculate_f1_score(
        predictions: List[str] | List[int],
        ground_truths: List[str] | List[int],
        positive_class: Optional[str] = None,
    ) -> MetricResult:
        """Calculate F1 score for simple binary classification settings."""
        if positive_class:
            pred_binary = [1 if isinstance(p, str) and positive_class in p else int(bool(p)) for p in predictions]
            truth_binary = [1 if isinstance(t, str) and positive_class in t else int(bool(t)) for t in ground_truths]
        else:
            pred_binary = [int(p) for p in predictions]  # type: ignore[arg-type]
            truth_binary = [int(t) for t in ground_truths]  # type: ignore[arg-type]

        tp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 0 and t == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(value=f1, details={"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn})

    @staticmethod
    def calculate_concept_coverage(
        responses: List[str],
        required_concepts: List[List[str]],
    ) -> MetricResult:
        """Check if responses cover required physics concepts."""
        if len(responses) != len(required_concepts):
            raise ValueError("Mismatched lengths")

        total_concepts = 0
        covered_concepts = 0
        samples = []

        for response, concepts in zip(responses, required_concepts):
            resp_l = response.lower()
            covered = [c for c in concepts if c.lower() in resp_l]
            total_concepts += len(concepts)
            covered_concepts += len(covered)
            coverage_rate = (len(covered) / len(concepts)) if concepts else 0.0
            samples.append({"response": response[:50], "required": concepts, "covered": covered, "coverage": coverage_rate})

        overall = (covered_concepts / total_concepts) if total_concepts > 0 else 0.0
        return MetricResult(value=overall, details={"covered": covered_concepts, "total": total_concepts, "samples": samples[:5]})

    @staticmethod
    def aggregate_metrics(metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
        """Aggregate multiple metrics into a summary with simple weights."""
        summary: Dict[str, Any] = {"overall_score": 0.0, "metrics": {}, "strengths": [], "weaknesses": []}
        weights = {"accuracy": 0.3, "unit_consistency": 0.2, "computation_accuracy": 0.25, "dimensional_analysis": 0.15, "concept_coverage": 0.1}

        weighted_sum = 0.0
        weight_total = 0.0

        for name, res in metrics.items():
            summary["metrics"][name] = {"value": res.value, "confidence": res.confidence}
            if name in weights:
                weighted_sum += res.value * weights[name]
                weight_total += weights[name]
            if res.value >= 0.8:
                summary["strengths"].append(f"{name}: {res.value:.1%}")
            elif res.value < 0.5:
                summary["weaknesses"].append(f"{name}: {res.value:.1%}")

        summary["overall_score"] = weighted_sum / weight_total if weight_total > 0 else 0.0
        return summary


# Convenience aggregator if you want a one-liner usage
def calculate_all_metrics(
    predictions: List[str],
    ground_truths: List[str],
    **kwargs: Any,
) -> Dict[str, MetricResult]:
    """Calculate a convenient bundle of standard metrics."""
    m = PhysicsMetrics()
    out: Dict[str, MetricResult] = {}

    out["accuracy"] = m.calculate_accuracy(predictions, ground_truths)
    out["fuzzy_accuracy"] = m.calculate_accuracy(predictions, ground_truths, method="fuzzy")
    out["unit_consistency"] = m.calculate_unit_consistency(predictions)

    if "numerical_truths" in kwargs:
        out["computation_accuracy"] = m.calculate_computation_accuracy(predictions, kwargs["numerical_truths"])

    if "expected_dimensions" in kwargs:
        out["dimensional_analysis"] = m.calculate_dimensional_analysis_score(predictions, kwargs["expected_dimensions"])

    if "required_concepts" in kwargs:
        out["concept_coverage"] = m.calculate_concept_coverage(predictions, kwargs["required_concepts"])

    return out
