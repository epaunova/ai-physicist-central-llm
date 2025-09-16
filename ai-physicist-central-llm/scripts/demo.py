#!/usr/bin/env python3
"""
Demo script for AI Physicist Central LLM
Run with: python scripts/demo.py
"""

import sys
from pathlib import Path

# Point Python to the project's src/ folder (no need for src package)
PROJECT_ROOT = Path(__file__).resolve().parents[1]      # .../ai-physicist-central-llm
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Sanity print (helps debugging if something is off)
print(f"[demo] Using SRC_DIR: {SRC_DIR}")
print(f"[demo] Exists: {SRC_DIR.exists()}  |  Contains: {', '.join(p.name for p in SRC_DIR.iterdir())}")

# Imports (NO 'src.' prefix here!)
from brain.specialized_model import PhysicsLLM
from hands.sympy_solver import SymPySolver
from hands.unit_checker import UnitChecker

def main():
    print("=" * 60)
    print("AI PHYSICIST DEMO")
    print("=" * 60)

    brain = PhysicsLLM()
    solver = SymPySolver()
    units = UnitChecker()
    print("✓ Components loaded")

    # 1) Pendulum
    q = "Calculate the period of a pendulum with length 2m on Earth"
    print("\nUser:", q)
    resp = brain.answer(q)
    print("Brain:", resp["answer"])

    calc = solver.calculate_pendulum_period(length=2.0)
    print("Hands[SymPy]:", calc)

    # 2) Kinetic Energy
    ke = solver.calculate_kinetic_energy(mass=5.0, velocity=10.0)
    print("\nKinetic Energy:", ke)

    # 3) Unit Conversion
    conv = units.convert_units(10.0, "m/s", "km/h")
    print("\nUnit Conversion:", conv)

    print("\n✓ Demo complete!")

if __name__ == "__main__":
    main()
