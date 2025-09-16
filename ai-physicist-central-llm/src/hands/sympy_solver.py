"""Symbolic mathematics solver using SymPy."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import sympy as sp


class SymPySolver:
    """Symbolic and numeric computations for common physics problems."""

    def __init__(self) -> None:
        self.constants: Dict[str, float] = {
            "g": 9.81,        # m/s²
            "c": 299_792_458, # m/s
            "h": 6.626e-34,   # J·s
            "k_e": 8.99e9,    # N·m²/C²
            "R": 8.314,       # J/(mol·K)
        }

    def solve_equation(
        self,
        equation: str,
        variable: str,
        values: Optional[Dict[str, Union[int, float, sp.Expr]]] = None,
    ) -> Dict[str, Any]:
        """
        Solve an equation for a given variable.

        Parameters
        ----------
        equation : str
            Either an 'Eq(lhs, rhs)' expression or any expression meaning 'expr = 0'.
        variable : str
            The symbol to solve for.
        values : dict, optional
            Substitutions (e.g., {"g": 9.81, "m": 2}).

        Returns
        -------
        dict
            {
              "success": bool,
              "solution": List[sympy.Expr] | None,
              "numeric": float | List[float] | None,
              "error": str | None
            }
        """
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(equation)

            # If equation is Eq(lhs, rhs), bring to lhs - rhs = 0
            if isinstance(expr, sp.Equality):
                expr = sp.simplify(expr.lhs - expr.rhs)

            # Otherwise assume expr == 0
            sol: List[sp.Expr] = sp.solve(sp.Eq(expr, 0), var)

            numeric: Optional[Union[float, List[float]]] = None
            if sol:
                if values:
                    try:
                        nums = [float(s.evalf(subs=values)) for s in sol]
                        numeric = nums if len(nums) > 1 else nums[0]
                    except Exception:
                        numeric = None

                return {
                    "success": True,
                    "solution": sol,
                    "numeric": numeric,
                }

            return {"success": True, "solution": [], "numeric": None}
        except Exception as e:
            return {"success": False, "error": str(e), "solution": None, "numeric": None}

    def calculate_pendulum_period(self, length: float) -> Dict[str, Any]:
        """Calculate the small-angle pendulum period: T = 2π√(L/g)."""
        try:
            if length <= 0:
                raise ValueError("length must be > 0")
            period = 2 * math.pi * math.sqrt(length / self.constants["g"])
            return {
                "success": True,
                "formula": "T = 2π√(L/g)",
                "result": round(period, 2),
                "unit": "s",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def calculate_kinetic_energy(self, mass: float, velocity: float) -> Dict[str, Any]:
        """Calculate kinetic energy: KE = ½ m v²."""
        try:
            if mass < 0:
                raise ValueError("mass must be ≥ 0")
            ke = 0.5 * mass * (velocity ** 2)
            return {
                "success": True,
                "formula": "KE = 1/2 m v^2",
                "result": float(ke),
                "unit": "J",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
