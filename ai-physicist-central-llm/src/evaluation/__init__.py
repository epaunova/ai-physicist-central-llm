cat > src/evaluation/__init__.py << 'EOF'
"""
Evaluation Module
=================
Comprehensive evaluation framework for physics QA models

This module provides:
- Metrics calculation (accuracy, unit consistency, etc.)
- Full evaluation pipeline
- Model comparison tools
- Report generation
"""

# Version
__version__ = "0.1.0"

# Core imports
from .metrics import (
    PhysicsMetrics,
    MetricResult,
    calculate_all_metrics
)

from .evaluator import (
    PhysicsEvaluator,
    EvaluationResult,
    Question,
    quick_evaluate
)

# Public API
__all__ = [
    # Metrics
    "PhysicsMetrics",
    "MetricResult",
    "calculate_all_metrics",
    
    # Evaluator
    "PhysicsEvaluator",
    "EvaluationResult",
    "Question",
    "quick_evaluate",
    
    # Module functions
    "run_evaluation",
    "compare_models",
    "load_dataset"
]

# Module-level configuration
DEFAULT_CONFIG = {
    "dataset_path": "data/evaluation/physics_qa_dataset.json",
    "output_dir": "data/results",
    "metrics": ["accuracy", "unit_consistency", "computation_accuracy"],
    "save_results": True,
    "verbose": True
}

# Import guard for optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting libraries not available. Install matplotlib and seaborn for visualizations.")


def run_evaluation(
    model,
    model_name: str = "Model",
    dataset_path: str = None,
    **kwargs
) -> EvaluationResult:
    """
    Run standard evaluation on a model
    
    Args:
        model: Model to evaluate (must have .answer() method)
        model_name: Name for identification
        dataset_path: Path to evaluation dataset
        **kwargs: Additional arguments for evaluator
        
    Returns:
        EvaluationResult object
        
    Example:
        >>> from src.evaluation import run_evaluation
        >>> from src.brain import PhysicsLLM
        >>> model = PhysicsLLM()
        >>> result = run_evaluation(model, "Physics LLM v1")
        >>> print(f"Accuracy: {result.accuracy:.1%}")
    """
    dataset_path = dataset_path or DEFAULT_CONFIG["dataset_path"]
    
    evaluator = PhysicsEvaluator(
        dataset_path=dataset_path,
        output_dir=kwargs.get("output_dir", DEFAULT_CONFIG["output_dir"])
    )
    
    return evaluator.evaluate(
        model=model,
        model_name=model_name,
        verbose=kwargs.get("verbose", DEFAULT_CONFIG["verbose"]),
        save_results=kwargs.get("save_results", DEFAULT_CONFIG["save_results"])
    )


def compare_models(models_list, dataset_path: str = None, **kwargs):
    """
    Compare multiple models
    
    Args:
        models_list: List of (model, name) tuples or (model, name, config) tuples
        dataset_path: Path to evaluation dataset
        **kwargs: Additional arguments
        
    Returns:
        Pandas DataFrame with comparison results
        
    Example:
        >>> from src.evaluation import compare_models
        >>> models = [
        ...     (baseline_model, "Baseline"),
        ...     (rag_model, "With RAG"),
        ...     (full_model, "RAG + Tools")
        ... ]
        >>> df = compare_models(models)
        >>> print(df)
    """
    dataset_path = dataset_path or DEFAULT_CONFIG["dataset_path"]
    
    evaluator = PhysicsEvaluator(dataset_path=dataset_path)
    
    # Normalize models list
    normalized_models = []
    for item in models_list:
        if len(item) == 2:
            model, name = item
            config = {}
        else:
            model, name, config = item
        normalized_models.append((model, name, config))
    
    return evaluator.compare_models(
        normalized_models,
        num_questions=kwargs.get("num_questions")
    )


def load_dataset(path: str = None) -> List[Question]:
    """
    Load evaluation dataset
    
    Args:
        path: Path to dataset file
        
    Returns:
        List of Question objects
        
    Example:
        >>> from src.evaluation import load_dataset
        >>> questions = load_dataset()
        >>> print(f"Loaded {len(questions)} questions")
    """
    path = path or DEFAULT_CONFIG["dataset_path"]
    evaluator = PhysicsEvaluator(dataset_path=path)
    return evaluator.questions


def plot_results(result: EvaluationResult, save_path: str = None):
    """
    Plot evaluation results
    
    Args:
        result: EvaluationResult to plot
        save_path: Optional path to save figure
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available")
        return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Evaluation Results: {result.model_name}", fontsize=16)
    
    # 1. Overall metrics
    ax1 = axes[0, 0]
    metrics = ['Accuracy', 'Unit Consistency', 'Computation']
    values = [result.accuracy, result.unit_consistency, result.computation_accuracy]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', fontweight='bold')
    
    # 2. By category
    ax2 = axes[0, 1]
    if result.by_category:
        categories = list(result.by_category.keys())
        cat_values = list(result.by_category.values())
        ax2.bar(categories, cat_values, color='#9b59b6')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Performance by Category')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. By question type
    ax3 = axes[1, 0]
    if result.by_type:
        types = list(result.by_type.keys())
        type_values = list(result.by_type.values())
        ax3.bar(types, type_values, color='#f39c12')
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Performance by Question Type')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Response time distribution
    ax4 = axes[1, 1]
    if result.response_times:
        ax4.hist(result.response_times, bins=20, color='#16a085', alpha=0.7)
        ax4.axvline(np.mean(result.response_times), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(result.response_times):.2f}s')
        ax4.set_xlabel('Response Time (s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Response Time Distribution')
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# Module initialization
def _init():
    """Initialize module"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

_init()
EOF
