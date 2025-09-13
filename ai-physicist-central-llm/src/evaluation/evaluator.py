cat > src/evaluation/evaluator.py << 'EOF'
"""
Evaluator Module
================
Main evaluation system for physics QA models
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
import numpy as np

from .metrics import PhysicsMetrics, MetricResult, calculate_all_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Single evaluation question"""
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
    """Complete evaluation results"""
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
    
    # Sample responses
    sample_responses: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error analysis
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Convert MetricResult objects
        if "all_metrics" in result:
            result["all_metrics"] = {
                k: {"value": v.value, "details": v.details}
                for k, v in self.all_metrics.items()
            }
        return result
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Save to JSON"""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str
    
    def summary(self) -> str:
        """Generate text summary"""
        summary = f"""
Evaluation Results for {self.model_name}
{'='*50}
Timestamp: {self.timestamp}
Questions Evaluated: {self.questions_evaluated}

Overall Performance:
  • Accuracy: {self.accuracy:.1%}
  • Unit Consistency: {self.unit_consistency:.1%}
  • Computation Accuracy: {self.computation_accuracy:.1%}

Performance by Category:
{self._format_breakdown(self.by_category)}

Performance by Type:
{self._format_breakdown(self.by_type)}

Performance by Difficulty:
{self._format_breakdown(self.by_difficulty)}
"""
        return summary
    
    def _format_breakdown(self, breakdown: Dict[str, float]) -> str:
        """Format breakdown dictionary"""
        lines = []
        for key, value in breakdown.items():
            lines.append(f"  • {key}: {value:.1%}")
        return '\n'.join(lines)


class PhysicsEvaluator:
    """
    Main evaluator for physics QA models
    
    This class handles the complete evaluation pipeline including:
    - Loading evaluation datasets
    - Running models on questions
    - Calculating metrics
    - Generating reports
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        metrics: Optional[PhysicsMetrics] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize evaluator
        
        Args:
            dataset_path: Path to evaluation dataset
            metrics: Metrics calculator (default: PhysicsMetrics)
            output_dir: Directory for saving results
        """
        self.dataset_path = dataset_path
        self.metrics = metrics or PhysicsMetrics()
        self.output_dir = Path(output_dir) if output_dir else Path("data/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.questions: List[Question] = []
        self.results_history: List[EvaluationResult] = []
        
        if dataset_path:
            self.load_dataset(dataset_path)
    
    def load_dataset(self, path: str) -> None:
        """Load evaluation dataset from JSON"""
        logger.info(f"Loading dataset from {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle different dataset formats
        if "physics_qa_dataset" in data:
            questions_data = data["physics_qa_dataset"]
        elif "questions" in data:
            questions_data = data["questions"]
        else:
            questions_data = data
        
        # Convert to Question objects
        self.questions = []
        for q_data in questions_data:
            if isinstance(q_data, dict):
                # Map fields appropriately
                question = Question(
                    id=q_data.get("id", len(self.questions)),
                    question=q_data.get("question", ""),
                    answer=q_data.get("answer", ""),
                    category=q_data.get("category", "general"),
                    type=q_data.get("type", "conceptual"),
                    difficulty=q_data.get("difficulty", "medium"),
                    solution=q_data.get("solution"),
                    units_required=q_data.get("units_required", []),
                    concepts=q_data.get("concepts", []),
                    numerical_answer=q_data.get("numerical_answer"),
                    dimension_answer=q_data.get("dimension_answer"),
                    metadata=q_data.get("metadata", {})
                )
                self.questions.append(question)
        
        logger.info(f"Loaded {len(self.questions)} questions")
        
        # Analyze dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self) -> None:
        """Analyze loaded dataset"""
        if not self.questions:
            return
        
        categories = set(q.category for q in self.questions)
        types = set(q.type for q in self.questions)
        difficulties = set(q.difficulty for q in self.questions)
        
        logger.info(f"Categories: {categories}")
        logger.info(f"Question types: {types}")
        logger.info(f"Difficulties: {difficulties}")
    
    def evaluate(
        self,
        model: Any,
        model_name: str = "Unknown Model",
        configuration: Optional[Dict[str, Any]] = None,
        num_questions: Optional[int] = None,
        verbose: bool = True,
        save_results: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a model on the loaded dataset
        
        Args:
            model: Model with .answer() method
            model_name: Name for identification
            configuration: Model configuration
            num_questions: Limit number of questions (None = all)
            verbose: Print progress
            save_results: Save results to file
            
        Returns:
            EvaluationResult object
        """
        if not self.questions:
            raise ValueError("No questions loaded")
        
        # Select questions
        questions_to_eval = self.questions[:num_questions] if num_questions else self.questions
        
        logger.info(f"Evaluating {model_name} on {len(questions_to_eval)} questions")
        
        # Storage for results
        predictions = []
        ground_truths = []
        response_times = []
        sample_responses = []
        
        # Additional data for metrics
        numerical_truths = []
        dimension_truths = []
        required_units = []
        required_concepts = []
        
        # Track by category/type
        results_by_category = {}
        results_by_type = {}
        results_by_difficulty = {}
        
        # Evaluate each question
        for i, question in enumerate(questions_to_eval):
            if verbose and i % 10 == 0:
                logger.info(f"Progress: {i}/{len(questions_to_eval)}")
            
            # Time the response
            start_time = time.time()
            
            try:
                # Get model response
                if hasattr(model, 'answer'):
                    response = model.answer(question.question)
                    if isinstance(response, dict):
                        prediction = response.get('answer', '')
                    else:
                        prediction = str(response)
                else:
                    # Fallback for simple models
                    prediction = model(question.question)
                    
            except Exception as e:
                logger.warning(f"Error on question {question.id}: {e}")
                prediction = ""
            
            elapsed = time.time() - start_time
            
            # Store results
            predictions.append(prediction)
            ground_truths.append(question.answer)
            response_times.append(elapsed)
            
            # Store additional data
            if question.numerical_answer is not None:
                numerical_truths.append(question.numerical_answer)
            if question.dimension_answer:
                dimension_truths.append(question.dimension_answer)
            if question.units_required:
                required_units.append(question.units_required)
            if question.concepts:
                required_concepts.append(question.concepts)
            
            # Track by category
            if question.category not in results_by_category:
                results_by_category[question.category] = {"correct": 0, "total": 0}
            results_by_category[question.category]["total"] += 1
            
            # Track by type
            if question.type not in results_by_type:
                results_by_type[question.type] = {"correct": 0, "total": 0}
            results_by_type[question.type]["total"] += 1
            
            # Track by difficulty
            if question.difficulty not in results_by_difficulty:
                results_by_difficulty[question.difficulty] = {"correct": 0, "total": 0}
            results_by_difficulty[question.difficulty]["total"] += 1
            
            # Check correctness
            is_correct = self._check_answer(prediction, question.answer)
            if is_correct:
                results_by_category[question.category]["correct"] += 1
                results_by_type[question.type]["correct"] += 1
                results_by_difficulty[question.difficulty]["correct"] += 1
            
            # Store sample responses
            if i < 10:  # First 10 as samples
                sample_responses.append({
                    "question": question.question,
                    "prediction": prediction,
                    "ground_truth": question.answer,
                    "correct": is_correct,
                    "time": elapsed
                })
        
        # Calculate all metrics
        logger.info("Calculating metrics...")
        
        metrics_results = {}
        
        # Basic accuracy
        metrics_results["accuracy"] = self.metrics.calculate_accuracy(
            predictions, ground_truths
        )
        metrics_results["fuzzy_accuracy"] = self.metrics.calculate_accuracy(
            predictions, ground_truths, method="fuzzy"
        )
        
        # Unit consistency
        metrics_results["unit_consistency"] = self.metrics.calculate_unit_consistency(
            predictions, required_units if required_units else None
        )
        
        # Computation accuracy (if numerical answers available)
        if numerical_truths:
            metrics_results["computation_accuracy"] = self.metrics.calculate_computation_accuracy(
                predictions[:len(numerical_truths)], numerical_truths
            )
        else:
            metrics_results["computation_accuracy"] = MetricResult(value=0.0)
        
        # Dimensional analysis (if available)
        if dimension_truths:
            metrics_results["dimensional_analysis"] = self.metrics.calculate_dimensional_analysis_score(
                predictions[:len(dimension_truths)], dimension_truths
            )
        
        # Concept coverage (if available)
        if required_concepts:
            metrics_results["concept_coverage"] = self.metrics.calculate_concept_coverage(
                predictions[:len(required_concepts)], required_concepts
            )
        
        # Response time analysis
        metrics_results["response_time"] = self.metrics.calculate_response_time_score(
            response_times
        )
        
        # Convert category/type results to percentages
        by_category_pct = {
            cat: data["correct"] / data["total"] if data["total"] > 0 else 0.0
            for cat, data in results_by_category.items()
        }
        by_type_pct = {
            typ: data["correct"] / data["total"] if data["total"] > 0 else 0.0
            for typ, data in results_by_type.items()
        }
        by_difficulty_pct = {
            diff: data["correct"] / data["total"] if data["total"] > 0 else 0.0
            for diff, data in results_by_difficulty.items()
        }
        
        # Error analysis
        error_analysis = self._analyze_errors(
            predictions, ground_truths, questions_to_eval
        )
        
        # Create result object
        result = EvaluationResult(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            configuration=configuration or {},
            accuracy=metrics_results["accuracy"].value,
            unit_consistency=metrics_results["unit_consistency"].value,
            computation_accuracy=metrics_results["computation_accuracy"].value,
            by_category=by_category_pct,
            by_type=by_type_pct,
            by_difficulty=by_difficulty_pct,
            questions_evaluated=len(questions_to_eval),
            correct_answers=sum(
                1 for p, g in zip(predictions, ground_truths) 
                if self._check_answer(p, g)
            ),
            response_times=response_times,
            all_metrics=metrics_results,
            sample_responses=sample_responses,
            error_analysis=error_analysis
        )
        
        # Save results
        if save_results:
            self._save_results(result)
        
        # Store in history
        self.results_history.append(result)
        
        # Print summary
        if verbose:
            print(result.summary())
        
        return result
    
    def _check_answer(self, prediction: str, ground_truth: str) -> bool:
        """Check if answer is correct"""
        # Simple check - can be made more sophisticated
        pred_clean = prediction.lower().strip()
        truth_clean = ground_truth.lower().strip()
        
        # Check if ground truth is contained in prediction
        if truth_clean in pred_clean:
            return True
        
        # Check numerical equivalence
        pred_nums = re.findall(r'[-+]?\d*\.?\d+', prediction)
        truth_nums = re.findall(r'[-+]?\d*\.?\d+', ground_truth)
        
        if pred_nums and truth_nums:
            try:
                pred_val = float(pred_nums[0])
                truth_val = float(truth_nums[0])
                if abs(pred_val - truth_val) / (abs(truth_val) + 1e-10) < 0.05:
                    return True
            except:
                pass
        
        return False
    
    def _analyze_errors(
        self,
        predictions: List[str],
        ground_truths: List[str],
        questions: List[Question]
    ) -> Dict[str, Any]:
        """Analyze error patterns"""
        errors = {
            "total_errors": 0,
            "error_types": {},
            "error_categories": {},
            "common_mistakes": []
        }
        
        for pred, truth, question in zip(predictions, ground_truths, questions):
            if not self._check_answer(pred, truth):
                errors["total_errors"] += 1
                
                # Categorize error
                error_type = self._categorize_error(pred, truth, question)
                errors["error_types"][error_type] = errors["error_types"].get(error_type, 0) + 1
                
                # Track by question category
                cat = question.category
                errors["error_categories"][cat] = errors["error_categories"].get(cat, 0) + 1
        
        return errors
    
    def _categorize_error(self, prediction: str, truth: str, question: Question) -> str:
        """Categorize type of error"""
        if not prediction:
            return "no_response"
        elif question.type == "numerical" and not re.findall(r'\d+', prediction):
            return "missing_calculation"
        elif question.units_required and not any(u in prediction for u in question.units_required):
            return "missing_units"
        elif len(prediction) < len(truth) / 2:
            return "incomplete_answer"
        else:
            return "incorrect_content"
    
    def _save_results(self, result: EvaluationResult) -> None:
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.model_name.replace(' ', '_')}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        result.to_json(str(filepath))
        logger.info(f"Results saved to {filepath}")
    
    def compare_models(
        self,
        models: List[Tuple[Any, str, Dict]],
        num_questions: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: List of (model, name, config) tuples
            num_questions: Number of questions to evaluate
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model, name, config in models:
            logger.info(f"Evaluating {name}...")
            result = self.evaluate(
                model, name, config, 
                num_questions=num_questions,
                verbose=False
            )
            
            comparison_data.append({
                "Model": name,
                "Accuracy": f"{result.accuracy:.1%}",
                "Unit Consistency": f"{result.unit_consistency:.1%}",
                "Computation": f"{result.computation_accuracy:.1%}",
                "Avg Response Time": f"{np.mean(result.response_times):.2f}s",
                "Questions": result.questions_evaluated
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def generate_report(
        self,
        result: Optional[EvaluationResult] = None,
        format: str = "markdown"
    ) -> str:
        """
        Generate evaluation report
        
        Args:
            result: Result to report (default: latest)
            format: "markdown", "html", or "latex"
            
        Returns:
            Formatted report string
        """
        if result is None:
            if not self.results_history:
                return "No results available"
            result = self.results_history[-1]
        
        if format == "markdown":
            return self._generate_markdown_report(result)
        elif format == "html":
            return self._generate_html_report(result)
        elif format == "latex":
            return self._generate_latex_report(result)
        else:
            return result.summary()
    
    def _generate_markdown_report(self, result: EvaluationResult) -> str:
        """Generate Markdown report"""
        report = f"""
# Evaluation Report: {result.model_name}

**Date**: {result.timestamp}  
**Questions Evaluated**: {result.questions_evaluated}

## Overall Performance

| Metric | Score |
|--------|-------|
| Accuracy | {result.accuracy:.1%} |
| Unit Consistency | {result.unit_consistency:.1%} |
| Computation Accuracy | {result.computation_accuracy:.1%} |

## Performance Breakdown

### By Category
"""
        for cat, score in result.by_category.items():
            report += f"- **{cat}**: {score:.1%}\n"
        
        report += "\n### By Question Type\n"
        for typ, score in result.by_type.items():
            report += f"- **{typ}**: {score:.1%}\n"
        
        report += "\n## Sample Responses\n\n"
        for i, sample in enumerate(result.sample_responses[:3], 1):
            report += f"**Q{i}**: {sample['question']}\n"
            report += f"- **Model**: {sample['prediction']}\n"
            report += f"- **Truth**: {sample['ground_truth']}\n"
            report += f"- **Correct**: {'✓' if sample['correct'] else '✗'}\n\n"
        
        return report
    
    def _generate_html_report(self, result: EvaluationResult) -> str:
        """Generate HTML report"""
        # Simplified HTML generation
        html = f"""
<!DOCTYPE html>
<html>
<head>
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
        return html
    
    def _generate_latex_report(self, result: EvaluationResult) -> str:
        """Generate LaTeX report"""
        latex = f"""
\\documentclass{{article}}
\\begin{{document}}

\\title{{Evaluation Report: {result.model_name}}}
\\date{{{result.timestamp}}}
\\maketitle

\\section{{Overall Performance}}

\\begin{{tabular}}{{|l|r|}}
\\hline
Metric & Score \\\\
\\hline
Accuracy & {result.accuracy:.1f}\\% \\\\
Unit Consistency & {result.unit_consistency:.1f}\\% \\\\
Computation Accuracy & {result.computation_accuracy:.1f}\\% \\\\
\\hline
\\end{{tabular}}

\\end{{document}}
"""
        return latex


# Convenience function
def quick_evaluate(model, dataset_path: str, model_name: str = "Model") -> EvaluationResult:
    """Quick evaluation function"""
    evaluator = PhysicsEvaluator(dataset_path)
    return evaluator.evaluate(model, model_name)
EOF
