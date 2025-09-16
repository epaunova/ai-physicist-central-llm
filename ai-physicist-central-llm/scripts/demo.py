import os
import sys
from pathlib import Path

# Add src/ to path (so imports like src.brain... work)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from src.brain.specialized_model import PhysicsLLM
from src.hands.sympy_solver import SymPySolver
from src.hands.unit_checker import UnitChecker


def main():
    brain, solver, units = PhysicsLLM(), SymPySolver(), UnitChecker()
    print("✓ Components loaded")

    # 1. Pendulum
    q = "Calculate the period of a pendulum with length 2m on Earth"
    print("\nUser:", q)
    resp = brain.answer(q)
    print("Brain:", resp["answer"])
    calc = solver.calculate_pendulum_period(length=2.0)
    print("Hands[Solver]:", calc)

    # 2. Kinetic Energy
    ke = solver.calculate_kinetic_energy(mass=5.0, velocity=10.0)
    print("\nKinetic Energy:", ke)

    # 3. Unit Conversion
    conv = units.convert_units(10.0, "m/s", "km/h")
    print("\nUnit Conversion:", conv)


if __name__ == "__main__":
    main()
