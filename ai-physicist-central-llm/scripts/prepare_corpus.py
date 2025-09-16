#!/usr/bin/env python3
"""
Corpus preparation stub for the AI Physicist project.

This script:
- Checks for a local JSON physics corpus at data/corpus/physics_abstracts.json
- Prints metadata and basic statistics if present
- Optionally bootstraps a tiny demo corpus with --init-demo (CPU-friendly)
- Outlines how a production-grade downloader would work (arXiv, cleaning, embeddings)

Usage:
  python scripts/prepare_corpus.py               # just report status
  python scripts/prepare_corpus.py --init-demo   # create a tiny demo corpus if missing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


# -------- Paths --------
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
CORPUS_PATH = CORPUS_DIR / "physics_abstracts.json"


def create_demo_corpus() -> List[Dict[str, str]]:
    """Return a tiny demo corpus (3 documents) suitable for local tests."""
    return [
        {
            "title": "Simple Pendulum Basics",
            "content": "For small angles, the period of a simple pendulum is T = 2π√(L/g). It depends on length L and gravity g."
        },
        {
            "title": "Coulomb's Law",
            "content": "The electric force between two charges is F = k_e q1 q2 / r^2. Electric field is force per unit charge."
        },
        {
            "title": "Heisenberg Uncertainty Principle",
            "content": "The uncertainty relation Δx Δp ≥ ħ/2 limits simultaneous precision of position and momentum."
        }
    ]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def report_corpus() -> Dict[str, str]:
    """Report corpus presence and basic metadata."""
    print("=" * 60)
    print("PHYSICS CORPUS PREPARATION (stub)")
    print("=" * 60)

    meta = {
        "source": "Local JSON (demo) or external sources (arXiv/textbooks)",
        "categories": [
            "classical_mechanics",
            "electromagnetism",
            "thermodynamics",
            "quantum_mechanics"
        ],
        "status": "absent" if not CORPUS_PATH.exists() else "ready"
    }

    if CORPUS_PATH.exists():
        data = load_json(CORPUS_PATH)
        n_docs = len(data)
        print(f"✔ Found corpus at: {CORPUS_PATH}")
        print(f"✔ Documents: {n_docs}")
        meta["documents_count"] = n_docs
    else:
        print(f"⚠ No corpus found at: {CORPUS_PATH}")
        meta["documents_count"] = 0

    print("\nProduction plan (outline):")
    print("  1) Fetch metadata via arXiv API (categories/keywords)")
    print("  2) Download abstracts / full text where permitted")
    print("  3) Clean & normalize text (remove LaTeX noise, deduplicate)")
    print("  4) Chunk documents and build embeddings (e.g., bge-small, E5)")
    print("  5) Persist to a vector store (e.g., FAISS) for retrieval")

    return meta


def maybe_init_demo() -> None:
    """Create a tiny demo corpus if missing."""
    if CORPUS_PATH.exists():
        print("Corpus already exists — skipping demo initialization.")
        return
    demo = create_demo_corpus()
    write_json(CORPUS_PATH, demo)
    print(f"✔ Demo corpus created: {CORPUS_PATH} (documents={len(demo)})")


def main():
    parser = argparse.ArgumentParser(description="Prepare/inspect physics corpus (stub).")
    parser.add_argument("--init-demo", action="store_true", help="Create a tiny demo corpus if missing.")
    args = parser.parse_args()

    meta = report_corpus()

    if args.init_demo:
        print("\n--init-demo requested")
        maybe_init_demo()
        # Re-report after creation
        print()
        report_corpus()

    print("\nDone.")


if __name__ == "__main__":
    main()
