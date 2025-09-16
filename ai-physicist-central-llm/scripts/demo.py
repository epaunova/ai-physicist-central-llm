#!/usr/bin/env python3
"""
Demo script for AI Physicist Central LLM
"""

import sys
from pathlib import Path

# Make sure src/ is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

# Import components
from brain.specialized_model import PhysicsLLM
from hands.sympy_solver import SymPySolver
from hands.unit_checker import UnitChecker
from knowledge.physics_corpus import load_default_corpus

def main():
    print("=" * 60)
    print(" AI PHYSICIST DEMO ")
    print("=" * 60)

    # Initialize components
    brain = PhysicsLLM()
    solver = SymPySolver()
    units = UnitChecker()
    corpus = load_default_corpus()
    print("✓ Components initialized")

    # 1. Simple question
    q1 = "What is Newton's second law?"
    print(f"\nUser: {q1}")
    resp1 = brain.answer(q1)
    print("Brain:", resp1.get("answer", resp1))

    # 2. Pendulum calculation
    q2 = "Calculate the period of a pendulum with length 2m on Earth"
    print(f"\nUser: {q2}")
    calc = solver.calculate_pendulum_period(length=2.0)
    print("Hands[Solver]:", calc)

    # 3. Kinetic energy
    ke = solver.calculate_kinetic_energy(mass=5.0, velocity=10.0)
    print("\nHands[Solver] Kinetic Energy:", ke)

    # 4. Unit conversion
    conv = units.convert_units(10.0, "m/s", "km/h")
    print("\nHands[UnitChecker] Conversion:", conv)

    # 5. Retrieval example
    q3 = "What is the uncertainty principle?"
    print(f"\nUser: {q3}")
    docs = corpus.search_documents(q3, limit=1)
    if docs:
        print("Knowledge Base:", docs[0].get_text())

    print("\n✓ Demo complete!")

if __name__ == "__main__":
    main()
