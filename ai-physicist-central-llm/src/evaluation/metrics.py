cat > src/evaluation/metrics.py << 'EOF'
"""
Evaluation Metrics Module
=========================
Comprehensive metrics for physics QA evaluation
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter
import re
from difflib import SequenceMatcher
import json


@dataclass
class MetricResult:
    """Container for metric results"""
    value: float
    details: Dict[str, Any] = None
    confidence: float = 1.0
    
    def __repr__(self):
        return f"MetricResult(value={self.value:.3f}, confidence={self.confidence:.3f})"


class PhysicsMetrics:
    """Comprehensive metrics for physics evaluation"""
    
    # Physical units patterns
    UNIT_PATTERNS = {
        'length': r'\b(m|meter|km|cm|mm)\b',
        'time': r'\b(s|second|min|hour|hr)\b',
        'mass': r'\b(kg|kilogram|g|gram|mg)\b',
        'force': r'\b(N|Newton|kN)\b',
        'energy': r'\b(J|Joule|kJ|eV|cal)\b',
        'power': r'\b(W|Watt|kW|MW)\b',
        'temperature': r'\b(K|Kelvin|°C|Celsius|°F)\b',
        'charge': r'\b(C|Coulomb|e)\b',
        'voltage': r'\b(V|Volt|kV|mV)\b',
        'current': r'\b(A|Ampere|mA)\b'
    }
    
    # Dimensional analysis patterns
    DIMENSION_PATTERNS = {
        'length': r'\[L\]',
        'mass': r'\[M\]',
        'time': r'\[T\]',
        'current': r'\[I\]',
        'temperature': r'\[Θ\]',
        'amount': r'\[N\]',
        'luminosity': r'\[J\]'
    }
    
    @staticmethod
    def calculate_accuracy(
        predictions: List[str], 
        ground_truths: List[str],
        method: str = "exact"
    ) -> MetricResult:
        """
        Calculate accuracy metric
        
        Args:
            predictions: Model predictions
            ground_truths: Ground truth answers
            method: "exact", "fuzzy", or "semantic"
            
        Returns:
            MetricResult with accuracy score
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        if not predictions:
            return MetricResult(value=0.0, details={"error": "No predictions"})
        
        correct = 0
        details = []
        
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
            details.append({"prediction": pred[:50], "truth": truth[:50], "correct": is_correct})
        
        accuracy = correct / len(predictions)
        
        return MetricResult(
            value=accuracy,
            details={
                "correct": correct,
                "total": len(predictions),
                "method": method,
                "samples": details[:5]  # First 5 samples
            }
        )
    
    @staticmethod
    def _fuzzy_match(pred: str, truth: str, threshold: float = 0.85) -> bool:
        """Fuzzy string matching"""
        pred_clean = re.sub(r'[^\w\s]', '', pred.lower())
        truth_clean = re.sub(r'[^\w\s]', '', truth.lower())
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, pred_clean, truth_clean).ratio()
        return similarity >= threshold
    
    @staticmethod
    def _semantic_match(pred: str, truth: str) -> bool:
        """Semantic matching for physics answers"""
        # Extract numbers from both
        pred_nums = re.findall(r'[-+]?\d*\.?\d+', pred)
        truth_nums = re.findall(r'[-+]?\d*\.?\d+', truth)
        
        # If both have numbers, check if they're close
        if pred_nums and truth_nums:
            try:
                pred_val = float(pred_nums[0])
                truth_val = float(truth_nums[0])
                # Allow 5% tolerance
                return abs(pred_val - truth_val) / (abs(truth_val) + 1e-10) < 0.05
            except:
                pass
        
        # Fallback to fuzzy matching
        return PhysicsMetrics._fuzzy_match(pred, truth, threshold=0.7)
    
    @staticmethod
    def calculate_unit_consistency(
        responses: List[str],
        expected_units: Optional[List[List[str]]] = None
    ) -> MetricResult:
        """
        Calculate unit consistency metric
        
        Args:
            responses: Model responses
            expected_units: Expected units for each response (optional)
            
        Returns:
            MetricResult with unit consistency score
        """
        total = len(responses)
        if total == 0:
            return MetricResult(value=0.0)
        
        correct = 0
        has_units = 0
        details = []
        
        for i, response in enumerate(responses):
            units_found = PhysicsMetrics._extract_units(response)
            
            if units_found:
                has_units += 1
                
                if expected_units and i < len(expected_units):
                    # Check if found units match expected
                    expected = expected_units[i]
                    if any(unit in response for unit in expected):
                        correct += 1
                        details.append({"response": response[:50], "units": units_found, "correct": True})
                else:
                    # Just check if units are present
                    correct += 1
            else:
                details.append({"response": response[:50], "units": None, "correct": False})
        
        consistency_score = correct / total if total > 0 else 0.0
        
        return MetricResult(
            value=consistency_score,
            details={
                "with_units": has_units,
                "correct_units": correct,
                "total": total,
                "unit_rate": has_units / total if total > 0 else 0,
                "samples": details[:5]
            }
        )
    
    @staticmethod
    def _extract_units(text: str) -> List[str]:
        """Extract physical units from text"""
        units_found = []
        
        for unit_type, pattern in PhysicsMetrics.UNIT_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                units_found.extend(matches)
        
        return list(set(units_found))
    
    @staticmethod
    def calculate_computation_accuracy(
        predictions: List[str],
        ground_truths: List[float],
        tolerance: float = 0.01
    ) -> MetricResult:
        """
        Calculate computational accuracy for numerical problems
        
        Args:
            predictions: Model predictions (may contain text)
            ground_truths: Numerical ground truth values
            tolerance: Relative tolerance for correctness
            
        Returns:
            MetricResult with computation accuracy
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Mismatched lengths")
        
        correct = 0
        extracted = 0
        details = []
        
        for pred, truth in zip(predictions, ground_truths):
            # Extract numerical value from prediction
            numbers = re.findall(r'[-+]?\d*\.?\d+', pred)
            
            if numbers:
                extracted += 1
                try:
                    pred_value = float(numbers[0])  # Take first number
                    
                    # Check if within tolerance
                    relative_error = abs(pred_value - truth) / (abs(truth) + 1e-10)
                    is_correct = relative_error <= tolerance
                    
                    if is_correct:
                        correct += 1
                    
                    details.append({
                        "predicted": pred_value,
                        "truth": truth,
                        "error": relative_error,
                        "correct": is_correct
                    })
                except:
                    details.append({"predicted": None, "truth": truth, "error": 1.0, "correct": False})
            else:
                details.append({"predicted": None, "truth": truth, "error": 1.0, "correct": False})
        
        accuracy = correct / len(predictions) if predictions else 0.0
        
        return MetricResult(
            value=accuracy,
            details={
                "correct": correct,
                "extracted": extracted,
                "total": len(predictions),
                "extraction_rate": extracted / len(predictions) if predictions else 0,
                "tolerance": tolerance,
                "samples": details[:5]
            }
        )
    
    @staticmethod
    def calculate_dimensional_analysis_score(
        responses: List[str],
        expected_dimensions: List[str]
    ) -> MetricResult:
        """
        Check dimensional analysis correctness
        
        Args:
            responses: Model responses
            expected_dimensions: Expected dimensional formulas
            
        Returns:
            MetricResult with dimensional analysis score
        """
        if len(responses) != len(expected_dimensions):
            raise ValueError("Mismatched lengths")
        
        correct = 0
        details = []
        
        for response, expected in zip(responses, expected_dimensions):
            # Extract dimensional expressions
            found_dims = re.findall(r'\[[\w\s\^\-\+\*\/]+\]', response)
            
            is_correct = False
            if found_dims:
                # Normalize and compare
                found_normalized = PhysicsMetrics._normalize_dimensions(found_dims[0])
                expected_normalized = PhysicsMetrics._normalize_dimensions(expected)
                is_correct = found_normalized == expected_normalized
            
            if is_correct:
                correct += 1
            
            details.append({
                "response": response[:50],
                "found": found_dims[0] if found_dims else None,
                "expected": expected,
                "correct": is_correct
            })
        
        score = correct / len(responses) if responses else 0.0
        
        return MetricResult(
            value=score,
            details={
                "correct": correct,
                "total": len(responses),
                "samples": details[:5]
            }
        )
    
    @staticmethod
    def _normalize_dimensions(dim_expr: str) -> str:
        """Normalize dimensional expression for comparison"""
        # Remove brackets and spaces
        normalized = re.sub(r'[\[\]\s]', '', dim_expr)
        
        # Sort terms for consistent comparison
        terms = re.findall(r'[A-Z][\^\-\d]*', normalized)
        terms.sort()
        
        return ''.join(terms)
    
    @staticmethod
    def calculate_response_time_score(
        response_times: List[float],
        max_acceptable_time: float = 5.0
    ) -> MetricResult:
        """
        Calculate response time score
        
        Args:
            response_times: List of response times in seconds
            max_acceptable_time: Maximum acceptable time
            
        Returns:
            MetricResult with time-based score
        """
        if not response_times:
            return MetricResult(value=0.0)
        
        avg_time = np.mean(response_times)
        median_time = np.median(response_times)
        
        # Score based on how many are under threshold
        under_threshold = sum(1 for t in response_times if t <= max_acceptable_time)
        score = under_threshold / len(response_times)
        
        return MetricResult(
            value=score,
            details={
                "avg_time": avg_time,
                "median_time": median_time,
                "min_time": min(response_times),
                "max_time": max(response_times),
                "under_threshold": under_threshold,
                "total": len(response_times),
                "threshold": max_acceptable_time
            }
        )
    
    @staticmethod
    def calculate_f1_score(
        predictions: List[str],
        ground_truths: List[str],
        positive_class: Optional[str] = None
    ) -> MetricResult:
        """Calculate F1 score for classification tasks"""
        # Convert to binary if needed
        if positive_class:
            pred_binary = [1 if positive_class in p else 0 for p in predictions]
            truth_binary = [1 if positive_class in t else 0 for t in ground_truths]
        else:
            # Assume already binary or will be converted
            pred_binary = predictions
            truth_binary = ground_truths
        
        # Calculate TP, FP, FN
        tp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 0 and t == 1)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return MetricResult(
            value=f1,
            details={
                "precision": precision,
                "recall": recall,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            }
        )
    
    @staticmethod
    def calculate_concept_coverage(
        responses: List[str],
        required_concepts: List[List[str]]
    ) -> MetricResult:
        """
        Check if responses cover required physics concepts
        
        Args:
            responses: Model responses
            required_concepts: List of concepts that should be mentioned
            
        Returns:
            MetricResult with concept coverage score
        """
        if len(responses) != len(required_concepts):
            raise ValueError("Mismatched lengths")
        
        total_concepts = 0
        covered_concepts = 0
        details = []
        
        for response, concepts in zip(responses, required_concepts):
            response_lower = response.lower()
            
            covered_in_response = []
            for concept in concepts:
                if concept.lower() in response_lower:
                    covered_concepts += 1
                    covered_in_response.append(concept)
                total_concepts += 1
            
            coverage_rate = len(covered_in_response) / len(concepts) if concepts else 0
            
            details.append({
                "response": response[:50],
                "required": concepts,
                "covered": covered_in_response,
                "coverage": coverage_rate
            })
        
        overall_coverage = covered_concepts / total_concepts if total_concepts > 0 else 0
        
        return MetricResult(
            value=overall_coverage,
            details={
                "covered": covered_concepts,
                "total": total_concepts,
                "samples": details[:5]
            }
        )
    
    @staticmethod
    def aggregate_metrics(metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
        """
        Aggregate multiple metrics into a summary
        
        Args:
            metrics: Dictionary of metric names to results
            
        Returns:
            Aggregated summary
        """
        summary = {
            "overall_score": 0.0,
            "metrics": {},
            "strengths": [],
            "weaknesses": []
        }
        
        # Calculate weighted average
        weights = {
            "accuracy": 0.3,
            "unit_consistency": 0.2,
            "computation_accuracy": 0.25,
            "dimensional_analysis": 0.15,
            "concept_coverage": 0.1
        }
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for name, result in metrics.items():
            summary["metrics"][name] = {
                "value": result.value,
                "confidence": result.confidence
            }
            
            # Add to weighted sum if weight exists
            if name in weights:
                weighted_sum += result.value * weights[name]
                weight_total += weights[name]
            
            # Identify strengths and weaknesses
            if result.value >= 0.8:
                summary["strengths"].append(f"{name}: {result.value:.1%}")
            elif result.value < 0.5:
                summary["weaknesses"].append(f"{name}: {result.value:.1%}")
        
        # Calculate overall score
        summary["overall_score"] = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        return summary


# Convenience functions
def calculate_all_metrics(
    predictions: List[str],
    ground_truths: List[str],
    **kwargs
) -> Dict[str, MetricResult]:
    """Calculate all standard metrics"""
    
    metrics = PhysicsMetrics()
    results = {}
    
    # Basic accuracy
    results["accuracy"] = metrics.calculate_accuracy(predictions, ground_truths)
    results["fuzzy_accuracy"] = metrics.calculate_accuracy(predictions, ground_truths, method="fuzzy")
    
    # Unit consistency
    results["unit_consistency"] = metrics.calculate_unit_consistency(predictions)
    
    # If numerical values provided
    if "numerical_truths" in kwargs:
        results["computation_accuracy"] = metrics.calculate_computation_accuracy(
            predictions, 
            kwargs["numerical_truths"]
        )
    
    # If dimensions provided
    if "expected_dimensions" in kwargs:
        results["dimensional_analysis"] = metrics.calculate_dimensional_analysis_score(
            predictions,
            kwargs["expected_dimensions"]
        )
    
    # If concepts provided
    if "required_concepts" in kwargs:
        results["concept_coverage"] = metrics.calculate_concept_coverage(
            predictions,
            kwargs["required_concepts"]
        )
    
    return results
EOF
