"""Symbolic mathematics solver using SymPy"""

import sympy as sp
from typing import Dict, Any, Optional
import math


class SymPySolver:
    """Symbolic computation for physics problems"""
    
    def __init__(self):
        self.constants = {
            'g': 9.81,  # m/s²
            'c': 299792458,  # m/s
            'h': 6.626e-34,  # J⋅s
            'k_e': 8.99e9,  # N⋅m²/C²
            'R': 8.314,  # J/(mol⋅K)
        }
        
    def solve_equation(self, equation: str, variable: str, values: Optional[Dict] = None) -> Dict[str, Any]:
        """Solve equation for given variable"""
        try:
            # Parse equation
            eq = sp.sympify(equation)
            var = sp.Symbol(variable)
            
            # Substitute values
            if values:
                eq = eq.subs(values)
                
            # Solve
            solution = sp.solve(eq, var)
            
            return {
                "success": True,
                "solution": solution,
                "numeric": float(solution[0]) if solution and len(solution) > 0 else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def calculate_pendulum_period(self, length: float) -> Dict[str, Any]:
        """Calculate pendulum period"""
        try:
            period = 2 * math.pi * math.sqrt(length / self.constants['g'])
            return {
                "success": True,
                "formula": "T = 2π√(L/g)",
                "result": round(period, 2),
                "unit": "seconds"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def calculate_kinetic_energy(self, mass: float, velocity: float) -> Dict[str, Any]:
        """Calculate kinetic energy"""
        try:
            ke = 0.5 * mass * velocity ** 2
            return {
                "success": True,
                "formula": "KE = ½mv²",
                "result": ke,
                "unit": "Joules"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
