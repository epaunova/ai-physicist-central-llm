# AI Physicist Central Language Model

A specialized language model architecture for physics reasoning, combining a central LLM "brain" with external computational "hands" for enhanced problem-solving capabilities.

## 🎯 Project Overview

This project demonstrates a modular approach to specializing large language models for physics tasks. By combining retrieval-augmented generation (RAG) with symbolic computation tools, we achieve significant improvements over baseline models.

### Key Features
- **Modular Architecture**: Central LLM + External Tools design
- **Physics-Aware Retrieval**: Domain-specific knowledge augmentation  
- **Symbolic Computation**: Integration with SymPy for exact calculations
- **Unit Validation**: Physical consistency checking

## 📊 Results

| Model Configuration | Accuracy | Unit Consistency |
|-------------------|----------|------------------|
| Baseline LLM | 42.3% | 31.2% |
| LLM + RAG | 58.7% | 45.3% |
| LLM + RAG + Tools | **71.2%** | **89.4%** |

## 🚀 Quick Start

Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

Run evaluation:
\`\`\`bash
python scripts/run_evaluation.py
\`\`\`
