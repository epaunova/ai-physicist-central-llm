"""
Demo script for AI Physicist components:
- PhysicsLLM (brain)
- SymPySolver (hands: math)
- UnitChecker (hands: units)

Run with: python scripts/demo.py
"""

import os
import sys
from pathlib import Path

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from brain.specialized_model import PhysicsLLM
from hands.sympy_solver import SymPySolver
from hands.unit_checker import UnitChecker


def main():
    brain, solver, units = PhysicsLLM(), SymPySolver(), UnitChecker()
    print("✓ Components loaded")

    # 1. Pendulum example
    q = "Calculate the period of a pendulum with length 2m on Earth"
    print("\nUser:", q)
    resp = brain.answer(q)
    print("Brain:", resp["answer"])
    calc = solver.calculate_pendulum_period(length=2.0)
    if calc["success"]:
        print("Hands[Solver]:", calc["result"], calc["unit"])

    # 2. Kinetic Energy example
    ke = solver.calculate_kinetic_energy(mass=5.0, velocity=10.0)
    if ke["success"]:
        print("\nKinetic Energy:", ke["result"], ke["unit"])

    # 3. Unit Conversion example
    conv = units.convert_units(10.0, "m/s", "km/h")
    if conv["success"]:
        print("\nUnit Conversion:", conv["original"], "->", conv["converted"])


if __name__ == "__main__":
    main()
