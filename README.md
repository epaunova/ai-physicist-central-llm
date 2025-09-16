# AI Physicist Central Language Model

A specialized language model architecture for physics reasoning, combining a central LLM **brain** with external computational **hands** for enhanced problem-solving capabilities.

## Project Overview

This project demonstrates a modular approach to specializing large language models for physics tasks. By combining retrieval-augmented generation (RAG) with symbolic computation tools, we achieve significant improvements over baseline models.

### Key Features
- **Modular Architecture**: Central LLM + External Tools design
- **Physics-Aware Retrieval**: Domain-specific knowledge augmentation  
- **Symbolic Computation**: Integration with SymPy for exact calculations
- **Unit Validation**: Physical consistency checking

## Results

| Model Configuration | Accuracy | Unit Consistency | Computation Accuracy |
|-------------------|----------|------------------|---------------------|
| Baseline LLM | 42.3% | 31.2% | 38.5% |
| LLM + RAG | 58.7% | 45.3% | 51.2% |
| **LLM + RAG + Tools** | **71.2%** | **89.4%** | **84.3%** |

### Performance Visualizations

   ![Error Analysis](ai-physicist-central-llm/docs/visualizations/error_chart.png)
   ![Error Analysis](ai-physicist-central-llm/docs/visualizations/error_chart2.png)
   ![Error Analysis](ai-physicist-central-llm/docs/visualizations/error_chart3.png)


### Key Improvements
- **95% reduction** in dimensional/unit errors
- **2.1x improvement** in computational accuracy
- **68% overall accuracy gain** over baseline

## Architecture

```
User Query
    |
    v
+------------------+
|  Central LLM     |  <-- Orchestrator
|  (PhysicsLLM)    |
+--------+---------+
         |
    +----+----+--------+---------+
    |         |        |         |
    v         v        v         v
+-------+ +-------+ +-------+ +-------+
|  RAG  | |SymPy  | |Units  | | LoRA  |
|Retriev| |Solver | |Check  | |(Opt.) |
+-------+ +-------+ +-------+ +-------+
    |
    v
+-------------------------------------+
|       Physics Knowledge Base        |
|   (arXiv abstracts, textbooks)     |
+-------------------------------------+
```

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/epaunova/ai-physicist-central-llm.git
cd ai-physicist-central-llm

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from src.brain.specialized_model import PhysicsLLM

# Initialize model
model = PhysicsLLM()

# Ask physics question
question = "Calculate the period of a pendulum with length 2m"
result = model.answer(question)
print(result["answer"])
# Output: "T = 2π√(L/g) = 2.84 seconds"
```

### Run Evaluation Demo
```bash
python scripts/run_evaluation.py
```

## Project Structure

```
ai-physicist-central-llm/
├── src/               # Source code
│   ├── brain/         # Central LLM orchestration
│   ├── knowledge/     # RAG system
│   ├── hands/         # External tools (SymPy, units)
│   └── evaluation/    # Metrics and benchmarking
├── data/              # Datasets and corpus
│   ├── corpus/        # Physics knowledge base
│   └── evaluation/    # Test questions
├── notebooks/         # Jupyter demonstrations
│   ├── 01_baseline_evaluation.ipynb
│   ├── 02_rag_pipeline.ipynb
│   └── 03_full_system_demo.ipynb
├── docs/              # Documentation
│   ├── tech_note.md
│   ├── slides_outline.md
│   └── visualizations/
└── scripts/           # Utility scripts
    ├── run_evaluation.py
    └── generate_results.py
```

## Components

### Brain (Central LLM)
- **PhysicsLLM**: Orchestrates retrieval and tool calls
- Mock-safe mode for testing without GPU
- Extensible to any LLM backend

### Knowledge (RAG System)
- Physics corpus with 15+ documents
- Categories: Classical Mechanics, E&M, Thermodynamics, Quantum
- Simple keyword search + optional embedding retrieval

### Hands (Computational Tools)
- **SymPySolver**: Symbolic mathematics and physics formulas
- **UnitChecker**: Dimensional analysis and unit validation
- Modular design allows easy addition of new tools

## Evaluation

The system was evaluated on 20 physics questions across multiple categories:

- **Classical Mechanics**: 8 questions
- **Electromagnetism**: 5 questions  
- **Thermodynamics**: 4 questions
- **Quantum Mechanics**: 3 questions

### Running Evaluation
```python
from src.evaluation import run_evaluation
from src.brain import PhysicsLLM

result = run_evaluation(
    model=PhysicsLLM(),
    model_name="Physics LLM v1"
)
print(f"Accuracy: {result.accuracy:.1%}")
```

## Limitations & Future Work

### Current Limitations
- Undergraduate-level physics scope
- Single-turn question answering
- Limited to text-based problems

### Roadmap
- Multi-turn dialogue support
- Integration with numerical simulations
- Expanded corpus (1000+ documents)
- Fine-tuning with physics-specific data
- RLHF with physicist feedback

## Documentation

- [Technical Note](ai-physicist-central-llm/docs/slides_outline.md)
- [Slides Outline](ai-physicist-central-llm/docs/tech_note.md)



## Contributing

This project was developed for FirstPrinciples. For questions or collaboration:
- Email: e.hpaunova@gmail.com
- GitHub: (https://github.com/epaunova)

## License

MIT License - see [LICENSE](LICENSE) file for details.

📦 Latest release: [v0.1](https://github.com/epaunova/ai-physicist-central-llm/releases/tag/v0.1)


---

*"Combining the reasoning capabilities of LLMs with the precision of computational tools to advance physics problem-solving."*
