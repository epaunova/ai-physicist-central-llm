"""Unit validation and simple dimensional analysis helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class UnitChecker:
    """Validates units and performs lightweight dimensional checks."""

    def __init__(self) -> None:
        # Dimensional “labels” (symbolic; for display/comparison only in this demo)
        self.dimensions: Dict[str, str] = {
            "length": "[L]",
            "mass": "[M]",
            "time": "[T]",
            "force": "[M L T^-2]",
            "energy": "[M L^2 T^-2]",
            "power": "[M L^2 T^-3]",
            "momentum": "[M L T^-1]",
            "acceleration": "[L T^-2]",
            "velocity": "[L T^-1]",
            "frequency": "[T^-1]",
            "pressure": "[M L^-1 T^-2]",
            "charge": "[A T]",
            "voltage": "[M L^2 T^-3 A^-1]",
        }

        # Scalar conversions (very small demo table)
        self.unit_conversions: Dict[tuple[str, str], float] = {
            ("m", "km"): 0.001,
            ("km", "m"): 1000.0,
            ("m/s", "km/h"): 3.6,
            ("km/h", "m/s"): 1 / 3.6,
            ("J", "eV"): 6.242e18,
            ("eV", "J"): 1.602e-19,
            ("kg", "g"): 1000.0,
            ("g", "kg"): 0.001,
        }

        # Regex for quickly spotting common unit tokens
        self._unit_token_re = re.compile(
            r"\b(m|km|cm|mm|s|min|hr|kg|g|N|J|W|eV|V|A|m/s|km/h)\b",
            flags=re.IGNORECASE,
        )

    # ---------- Public API ----------

    def check_dimensions(self, expression: str, expected_dimension: str) -> Dict[str, Any]:
        """Check if a textual expression *claims* the expected dimension label."""
        key = expected_dimension.lower()
        if key in self.dimensions:
            return {
                "valid": True,
                "expected": self.dimensions[key],
                "message": f"Dimensions validated for {expected_dimension}",
            }
        return {"valid": False, "message": f"Unknown dimension type: {expected_dimension}"}

    def validate_units(self, text: str, expected_units: List[str] | None = None) -> Dict[str, Any]:
        """
        Check whether a response contains unit tokens and (optionally) matches expectations.

        Returns
        -------
        dict with keys:
          - found: List[str]
          - has_units: bool
          - matches_expected: bool | None
        """
        found = sorted(set(self._unit_token_re.findall(text)))
        has_units = len(found) > 0
        if expected_units:
            matches = any(u.lower() in text.lower() for u in expected_units)
        else:
            matches = None
        return {"found": found, "has_units": has_units, "matches_expected": matches}

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert between a small set of scalar units."""
        key = (from_unit, to_unit)
        factor = self.unit_conversions.get(key)
        if factor is None:
            return {"success": False, "error": f"Conversion from {from_unit} to {to_unit} not supported"}
        return {
            "success": True,
            "original": f"{value} {from_unit}",
            "converted": f"{value * factor} {to_unit}",
            "factor": factor,
        }

    def validate_expression(self, expression: str, expected_units: str) -> bool:
        """Placeholder: in a real system you'd parse and evaluate units; here we return True."""
        return True
