"""
Evaluator Module
================
Main evaluation system for physics QA models.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import PhysicsMetrics, MetricResult

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Single evaluation question."""
    id: int
    question: str
    answer: str
    category: str = "general"
    type: str = "conceptual"
    difficulty: str = "medium"
    solution: Optional[str] = None
    units_required: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    numerical_answer: Optional[float] = None
    dimension_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    model_name: str
    timestamp: str
    configuration: Dict[str, Any]

    # Overall metrics
    accuracy: float
    unit_consistency: float
    computation_accuracy: float

    # Detailed breakdowns
    by_category: Dict[str, float]
    by_type: Dict[str, float]
    by_difficulty: Dict[str, float]

    # Raw results
    questions_evaluated: int
    correct_answers: int
    response_times: List[float]

    # Detailed metrics
    all_metrics: Dict[str, MetricResult]

    # Samples and errors
    sample_responses: List[Dict[str, Any]] = field(default_factory=list)
    error_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        data = asdict(self)
        # Convert MetricResult objects
        if "all_metrics" in data and isinstance(self.all_metrics, dict):
            data["all_metrics"] = {k: {"value": v.value, "details": v.details} for k, v in self.all_metrics.items()}
        return data

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to JSON (optionally write to disk)."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def _format_breakdown(self, breakdown: Dict[str, float]) -> str:
        return "\n".join(f"  • {k}: {v:.1%}" for k, v in breakdown.items())

    def summary(self) -> str:
        """Text summary for console output."""
        return (
            f"\nEvaluation Results for {self.model_name}\n"
            f"{'=' * 50}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Questions Evaluated: {self.questions_evaluated}\n\n"
            f"Overall Performance:\n"
            f"  • Accuracy: {self.accuracy:.1%}\n"
            f"  • Unit Consistency: {self.unit_consistency:.1%}\n"
            f"  • Computation Accuracy: {self.computation_accuracy:.1%}\n\n"
            f"Performance by Category:\n{self._format_breakdown(self.by_category)}\n\n"
            f"Performance by Type:\n{self._format_breakdown(self.by_type)}\n\n"
            f"Performance by Difficulty:\n{self._format_breakdown(self.by_difficulty)}\n"
        )


class PhysicsEvaluator:
    """
    Main evaluator for physics QA models.

    Pipeline:
    - Load evaluation dataset
    - Run the model on questions
    - Compute metrics and breakdowns
    - Save results and generate reports
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        metrics: Optional[PhysicsMetrics] = None,
        output_dir: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.metrics = metrics or PhysicsMetrics()
        self.output_dir = Path(output_dir) if output_dir else Path("data/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.questions: List[Question] = []
        self.results_history: List[EvaluationResult] = []

        if dataset_path:
            self.load_dataset(dataset_path)

    def load_dataset(self, path: str) -> None:
        """Load evaluation dataset from JSON."""
        logger.info(f"Loading dataset from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "physics_qa_dataset" in data:
            questions_data = data["physics_qa_dataset"]
        elif "questions" in data:
            questions_data = data["questions"]
        else:
            questions_data = data

        self.questions = []
        for idx, qd in enumerate(questions_data):
            if isinstance(qd, dict):
                self.questions.append(
                    Question(
                        id=qd.get("id", idx),
                        question=qd.get("question", ""),
                        answer=qd.get("answer", ""),
                        category=qd.get("category", "general"),
                        type=qd.get("type", "conceptual"),
                        difficulty=qd.get("difficulty", "medium"),
                        solution=qd.get("solution"),
                        units_required=qd.get("units_required", []),
                        concepts=qd.get("concepts", []),
                        numerical_answer=qd.get("numerical_answer"),
                        dimension_answer=qd.get("dimension_answer"),
                        metadata=qd.get("metadata", {}),
                    )
                )

        logger.info(f"Loaded {len(self.questions)} questions")
        self._analyze_dataset()

    def _analyze_dataset(self) -> None:
        """Log basic dataset composition."""
        if not self.questions:
            return
        categories = sorted({q.category for q in self.questions})
        types = sorted({q.type for q in self.questions})
        difficulties = sorted({q.difficulty for q in self.questions})
        logger.info(f"Categories: {categories}")
        logger.info(f"Types: {types}")
        logger.info(f"Difficulties: {difficulties}")

    def evaluate(
        self,
        model: Any,
        model_name: str = "Unknown Model",
        configuration: Optional[Dict[str, Any]] = None,
        num_questions: Optional[int] = None,
        verbose: bool = True,
        save_results: bool = True,
    ) -> EvaluationResult:
        """Evaluate a model on the loaded dataset."""
        if not self.questions:
            raise ValueError("No questions loaded")

        items = self.questions[: num_questions] if num_questions else self.questions
        logger.info(f"Evaluating {model_name} on {len(items)} questions")

        predictions: List[str] = []
        truths: List[str] = []
        response_times: List[float] = []
        samples: List[Dict[str, Any]] = []

        numerical_truths: List[float] = []
        dimension_truths: List[str] = []
        required_units: List[List[str]] = []
        required_concepts: List[List[str]] = []

        by_category: Dict[str, Dict[str, int]] = {}
        by_type: Dict[str, Dict[str, int]] = {}
        by_diff: Dict[str, Dict[str, int]] = {}

        for i, q in enumerate(items):
            if verbose and i % 10 == 0:
                logger.info(f"Progress: {i}/{len(items)}")

            t0 = time.time()
            try:
                if hasattr(model, "answer"):
                    resp = model.answer(q.question)
                    pred = resp.get("answer", "") if isinstance(resp, dict) else str(resp)
                else:
                    pred = str(model(q.question))
            except Exception as e:
                logger.warning(f"Error on question {q.id}: {e}")
                pred = ""
            dt = time.time() - t0

            predictions.append(pred)
            truths.append(q.answer)
            response_times.append(dt)

            if q.numerical_answer is not None:
                numerical_truths.append(q.numerical_answer)
            if q.dimension_answer:
                dimension_truths.append(q.dimension_answer)
            if q.units_required:
                required_units.append(q.units_required)
            if q.concepts:
                required_concepts.append(q.concepts)

            by_category.setdefault(q.category, {"correct": 0, "total": 0})["total"] += 1
            by_type.setdefault(q.type, {"correct": 0, "total": 0})["total"] += 1
            by_diff.setdefault(q.difficulty, {"correct": 0, "total": 0})["total"] += 1

            ok = self._check_answer(pred, q.answer)
            if ok:
                by_category[q.category]["correct"] += 1
                by_type[q.type]["correct"] += 1
                by_diff[q.difficulty]["correct"] += 1

            if i < 10:
                samples.append(
                    {
                        "question": q.question,
                        "prediction": pred,
                        "ground_truth": q.answer,
                        "correct": ok,
                        "time": dt,
                    }
                )

        # Metrics
        logger.info("Calculating metrics...")
        m = self.metrics

        metrics_results: Dict[str, MetricResult] = {}
        metrics_results["accuracy"] = m.calculate_accuracy(predictions, truths)
        metrics_results["fuzzy_accuracy"] = m.calculate_accuracy(predictions, truths, method="fuzzy")
        metrics_results["unit_consistency"] = m.calculate_unit_consistency(predictions, required_units or None)
        if numerical_truths:
            metrics_results["computation_accuracy"] = m.calculate_computation_accuracy(
                predictions[: len(numerical_truths)], numerical_truths
            )
        else:
            metrics_results["computation_accuracy"] = MetricResult(value=0.0)
        if dimension_truths:
            metrics_results["dimensional_analysis"] = m.calculate_dimensional_analysis_score(
                predictions[: len(dimension_truths)], dimension_truths
            )
        if required_concepts:
            metrics_results["concept_coverage"] = m.calculate_concept_coverage(
                predictions[: len(required_concepts)], required_concepts
            )
        metrics_results["response_time"] = m.calculate_response_time_score(response_times)

        by_category_pct = {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in by_category.items()}
        by_type_pct = {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in by_type.items()}
        by_diff_pct = {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in by_diff.items()}

        errors = self._analyze_errors(predictions, truths, items)

        result = EvaluationResult(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            configuration=configuration or {},
            accuracy=metrics_results["accuracy"].value,
            unit_consistency=metrics_results["unit_consistency"].value,
            computation_accuracy=metrics_results["computation_accuracy"].value,
            by_category=by_category_pct,
            by_type=by_type_pct,
            by_difficulty=by_diff_pct,
            questions_evaluated=len(items),
            correct_answers=sum(1 for p, g in zip(predictions, truths) if self._check_answer(p, g)),
            response_times=response_times,
            all_metrics=metrics_results,
            sample_responses=samples,
            error_analysis=errors,
        )

        if save_results:
            self._save_results(result)

        self.results_history.append(result)

        if verbose:
            print(result.summary())

        return result

    def _check_answer(self, prediction: str, ground_truth: str) -> bool:
        """Heuristic correctness check (exact/contains or close numeric)."""
        pred_clean = prediction.lower().strip()
        truth_clean = ground_truth.lower().strip()
        if truth_clean and truth_clean in pred_clean:
            return True

        pred_nums = re.findall(r"[-+]?\d*\.?\d+", prediction)
        truth_nums = re.findall(r"[-+]?\d*\.?\d+", ground_truth)
        if pred_nums and truth_nums:
            try:
                pval = float(pred_nums[0])
                tval = float(truth_nums[0])
                return abs(pval - tval) / (abs(tval) + 1e-10) < 0.05
            except Exception:
                return False
        return False

    def _analyze_errors(
        self,
        predictions: List[str],
        ground_truths: List[str],
        questions: List[Question],
    ) -> Dict[str, Any]:
        """Categorize simple error patterns for reporting."""
        errors: Dict[str, Any] = {"total_errors": 0, "error_types": {}, "error_categories": {}, "common_mistakes": []}

        for pred, truth, q in zip(predictions, ground_truths, questions):
            if self._check_answer(pred, truth):
                continue
            errors["total_errors"] += 1
            etype = self._categorize_error(pred, truth, q)
            errors["error_types"][etype] = errors["error_types"].get(etype, 0) + 1
            errors["error_categories"][q.category] = errors["error_categories"].get(q.category, 0) + 1

        return errors

    def _categorize_error(self, prediction: str, truth: str, question: Question) -> str:
        if not prediction:
            return "no_response"
        if question.type == "numerical" and not re.findall(r"\d+", prediction):
            return "missing_calculation"
        if question.units_required and not any(u in prediction for u in question.units_required):
            return "missing_units"
        if len(prediction) < max(1, len(truth) // 2):
            return "incomplete_answer"
        return "incorrect_content"

    def _save_results(self, result: EvaluationResult) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{result.model_name.replace(' ', '_')}_{ts}.json"
        fpath = self.output_dir / fname
        result.to_json(str(fpath))
        logger.info(f"Results saved to {fpath}")

    def compare_models(
        self,
        models: List[Tuple[Any, str, Dict[str, Any]]],
        num_questions: Optional[int] = None,
    ) -> pd.DataFrame:
        """Evaluate and compare multiple models; return a DataFrame."""
        rows: List[Dict[str, Any]] = []
        for model, name, config in models:
            logger.info(f"Evaluating {name}...")
            res = self.evaluate(model, name, config, num_questions=num_questions, verbose=False, save_results=False)
            rows.append(
                {
                    "Model": name,
                    "Accuracy": f"{res.accuracy:.1%}",
                    "Unit Consistency": f"{res.unit_consistency:.1%}",
                    "Computation": f"{res.computation_accuracy:.1%}",
                    "Avg Response Time": f"{float(np.mean(res.response_times)):.2f}s" if res.response_times else "n/a",
                    "Questions": res.questions_evaluated,
                }
            )
        return pd.DataFrame(rows)

    def generate_report(self, result: Optional[EvaluationResult] = None, format: str = "markdown") -> str:
        """Generate a simple report in markdown/html/latex or text summary."""
        if result is None:
            if not self.results_history:
                return "No results available"
            result = self.results_history[-1]

        if format == "markdown":
            return self._generate_markdown_report(result)
        if format == "html":
            return self._generate_html_report(result)
        if format == "latex":
            return self._generate_latex_report(result)
        return result.summary()

    def _generate_markdown_report(self, result: EvaluationResult) -> str:
        lines = [
            f"# Evaluation Report: {result.model_name}",
            "",
            f"**Date**: {result.timestamp}  ",
            f"**Questions Evaluated**: {result.questions_evaluated}",
            "",
            "## Overall Performance",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Accuracy | {result.accuracy:.1%} |",
            f"| Unit Consistency | {result.unit_consistency:.1%} |",
            f"| Computation Accuracy | {result.computation_accuracy:.1%} |",
            "",
            "## Performance Breakdown",
            "",
            "### By Category",
        ]
        for cat, score in result.by_category.items():
            lines.append(f"- **{cat}**: {score:.1%}")
        lines.append("")
        lines.append("### By Question Type")
        for typ, score in result.by_type.items():
            lines.append(f"- **{typ}**: {score:.1%}")
        lines.append("")
        lines.append("## Sample Responses")
        lines.append("")
        for i, sample in enumerate(result.sample_responses[:3], 1):
            lines.append(f"**Q{i}**: {sample['question']}")
            lines.append(f"- **Model**: {sample['prediction']}")
            lines.append(f"- **Truth**: {sample['ground_truth']}")
            lines.append(f"- **Correct**: {'✓' if sample['correct'] else '✗'}")
            lines.append("")
        return "\n".join(lines)

    def _generate_html_report(self, result: EvaluationResult) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Evaluation Report: {result.model_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #4CAF50; color: white; }}
    .correct {{ color: green; }}
    .incorrect {{ color: red; }}
  </style>
</head>
<body>
  <h1>Evaluation Report: {result.model_name}</h1>
  <p><strong>Date:</strong> {result.timestamp}</p>
  <p><strong>Questions:</strong> {result.questions_evaluated}</p>

  <h2>Overall Performance</h2>
  <table>
    <tr><th>Metric</th><th>Score</th></tr>
    <tr><td>Accuracy</td><td>{result.accuracy:.1%}</td></tr>
    <tr><td>Unit Consistency</td><td>{result.unit_consistency:.1%}</td></tr>
    <tr><td>Computation Accuracy</td><td>{result.computation_accuracy:.1%}</td></tr>
  </table>
</body>
</html>
"""

    def _generate_latex_report(self, result: EvaluationResult) -> str:
        return rf"""\documentclass{{article}}
\begin{document}
\title{{Evaluation Report: {result.model_name}}}
\date{{{result.timestamp}}}
\maketitle

\section*{{Overall Performance}}
\begin{{tabular}}{{|l|r|}}
\hline
Metric & Score \\
\hline
Accuracy & {result.accuracy:.1f}\% \\
Unit Consistency & {result.unit_consistency:.1f}\% \\
Computation Accuracy & {result.computation_accuracy:.1f}\% \\
\hline
\end{{tabular}}
\end{document}
"""


# Convenience function
def quick_evaluate(model: Any, dataset_path: str, model_name: str = "Model") -> EvaluationResult:
    evaluator = PhysicsEvaluator(dataset_path)
    return evaluator.evaluate(model, model_name)
