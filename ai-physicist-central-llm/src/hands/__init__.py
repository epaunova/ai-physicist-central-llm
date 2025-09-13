cat > src/hands/__init__.py << 'EOF'
"""
Hands Module - External Computational Tools
============================================

This module provides computational tools for physics calculations:
- SymPy solver for symbolic mathematics
- Unit checker for dimensional analysis
- Numerical solver for computational physics
- Formula parser and evaluator
"""

__version__ = "0.1.0"

# Core imports
from .sympy_solver import SymPySolver
from .unit_checker import UnitChecker

# Public API
__all__ = [
    "SymPySolver",
    "UnitChecker",
    "ToolRegistry",
    "PhysicsCalculator",
    "solve_physics_problem",
    "check_units",
    "evaluate_formula"
]

# Tool registry for dynamic loading
class ToolRegistry:
    """Registry for computational tools"""
    
    _tools = {}
    _instances = {}
    
    @classmethod
    def register(cls, name: str, tool_class):
        """Register a new tool"""
        cls._tools[name] = tool_class
    
    @classmethod
    def get_tool(cls, name: str):
        """Get tool instance (singleton)"""
        if name not in cls._instances:
            if name not in cls._tools:
                raise ValueError(f"Unknown tool: {name}")
            cls._instances[name] = cls._tools[name]()
        return cls._instances[name]
    
    @classmethod
    def list_tools(cls):
        """List available tools"""
        return list(cls._tools.keys())
    
    @classmethod
    def get_all_tools(cls):
        """Get all tool instances"""
        return {
            name: cls.get_tool(name) 
            for name in cls._tools.keys()
        }


# Register default tools
ToolRegistry.register("sympy", SymPySolver)
ToolRegistry.register("units", UnitChecker)


# Combined physics calculator
class PhysicsCalculator:
    """
    Unified interface for physics calculations
    
    Combines all computational tools into a single interface
    """
    
    def __init__(self):
        """Initialize with all available tools"""
        self.solver = SymPySolver()
        self.unit_checker = UnitChecker()
        self.tools = ToolRegistry.get_all_tools()
    
    def solve(self, problem: str, **kwargs):
        """
        Solve a physics problem
        
        Args:
            problem: Problem description or equation
            **kwargs: Additional parameters
            
        Returns:
            Solution dictionary
        """
        result = {
            "problem": problem,
            "solution": None,
            "units": None,
            "method": None,
            "confidence": 0.0
        }
        
        # Detect problem type
        problem_lower = problem.lower()
        
        # Try symbolic solution
        if any(symbol in problem for symbol in ['=', 'solve', 'find']):
            try:
                # Extract equation
                if '=' in problem:
                    equation = problem.split('solve')[-1].strip()
                    if 'for' in equation:
                        eq_part, var_part = equation.split('for')
                        equation = eq_part.strip()
                        variable = var_part.strip()
                    else:
                        equation = equation.strip()
                        variable = 'x'
                    
                    sol = self.solver.solve_equation(equation, variable)
                    if sol["success"]:
                        result["solution"] = sol["solution"]
                        result["method"] = "symbolic"
                        result["confidence"] = 0.9
            except Exception as e:
                logger.warning(f"Symbolic solution failed: {e}")
        
        # Check for specific physics problems
        if "pendulum" in problem_lower:
            if "period" in problem_lower:
                # Extract length
                import re
                numbers = re.findall(r'[\d.]+', problem)
                if numbers:
                    length = float(numbers[0])
                    sol = self.solver.calculate_pendulum_period(length)
                    if sol["success"]:
                        result["solution"] = sol["result"]
                        result["units"] = sol["unit"]
                        result["method"] = "pendulum_formula"
                        result["confidence"] = 0.95
        
        elif "kinetic" in problem_lower and "energy" in problem_lower:
            # Extract mass and velocity
            import re
            numbers = re.findall(r'[\d.]+', problem)
            if len(numbers) >= 2:
                mass = float(numbers[0])
                velocity = float(numbers[1])
                sol = self.solver.calculate_kinetic_energy(mass, velocity)
                if sol["success"]:
                    result["solution"] = sol["result"]
                    result["units"] = sol["unit"]
                    result["method"] = "kinetic_energy"
                    result["confidence"] = 0.95
        
        # Validate units if solution found
        if result["solution"] and result["units"]:
            unit_check = self.unit_checker.validate_expression(
                str(result["solution"]) + " " + result["units"],
                problem
            )
            if not unit_check:
                result["confidence"] *= 0.8
        
        return result
    
    def check_dimensional_consistency(self, equation: str) -> bool:
        """Check if equation is dimensionally consistent"""
        return self.unit_checker.check_equation_dimensions(equation)
    
    def convert(self, value: float, from_unit: str, to_unit: str):
        """Convert between units"""
        return self.unit_checker.convert_units(value, from_unit, to_unit)
    
    def evaluate(self, formula: str, variables: dict):
        """Evaluate formula with given variables"""
        return self.solver.evaluate_expression(formula, variables)


# Convenience functions
def solve_physics_problem(problem: str, **kwargs):
    """
    Quick function to solve physics problems
    
    Args:
        problem: Problem description
        **kwargs: Additional parameters
        
    Returns:
        Solution dictionary
        
    Example:
        >>> solve_physics_problem("Calculate pendulum period for L=2m")
        {'solution': 2.84, 'units': 'seconds', ...}
    """
    calculator = PhysicsCalculator()
    return calculator.solve(problem, **kwargs)


def check_units(expression: str, expected_dimension: str = None):
    """
    Check units in expression
    
    Args:
        expression: Expression to check
        expected_dimension: Expected dimensional formula
        
    Returns:
        Validation result
        
    Example:
        >>> check_units("10 kg * 5 m/s^2", "force")
        {'valid': True, 'dimension': '[M L T^-2]'}
    """
    checker = UnitChecker()
    if expected_dimension:
        return checker.check_dimensions(expression, expected_dimension)
    else:
        return checker.extract_units(expression)


def evaluate_formula(formula: str, **variables):
    """
    Evaluate physics formula
    
    Args:
        formula: Formula string
        **variables: Variable values
        
    Returns:
        Calculated result
        
    Example:
        >>> evaluate_formula("F = m * a", m=10, a=5)
        50
    """
    solver = SymPySolver()
    return solver.evaluate_expression(formula, variables)


# Physics constants
class PhysicsConstants:
    """Common physics constants"""
    
    # Fundamental constants
    c = 299792458  # Speed of light (m/s)
    h = 6.62607015e-34  # Planck constant (J⋅s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J⋅s)
    G = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
    e = 1.602176634e-19  # Elementary charge (C)
    me = 9.1093837015e-31  # Electron mass (kg)
    mp = 1.67262192369e-27  # Proton mass (kg)
    mn = 1.67492749804e-27  # Neutron mass (kg)
    k = 8.9875517923e9  # Coulomb constant (N⋅m²/C²)
    epsilon0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
    mu0 = 1.25663706212e-6  # Vacuum permeability (H/m)
    
    # Thermodynamic constants
    kB = 1.380649e-23  # Boltzmann constant (J/K)
    R = 8.314462618  # Gas constant (J/mol⋅K)
    NA = 6.02214076e23  # Avogadro's number (1/mol)
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²⋅K⁴)
    
    # Earth constants
    g = 9.80665  # Standard gravity (m/s²)
    atm = 101325  # Standard atmosphere (Pa)
    
    @classmethod
    def get_all(cls):
        """Get all constants as dictionary"""
        return {
            name: value 
            for name, value in cls.__dict__.items() 
            if not name.startswith('_') and not callable(value)
        }
    
    @classmethod
    def get(cls, name: str, default=None):
        """Get constant by name"""
        return getattr(cls, name, default)


# Tool configuration
TOOL_CONFIG = {
    "sympy": {
        "precision": 10,
        "timeout": 5.0,
        "simplify": True
    },
    "units": {
        "strict_mode": False,
        "allow_implicit": True,
        "default_system": "SI"
    },
    "numerical": {
        "method": "scipy",
        "tolerance": 1e-10,
        "max_iterations": 1000
    }
}


# Module initialization
def _init_module():
    """Initialize hands module"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Check for optional dependencies
    try:
        import sympy
        logger.info("SymPy available for symbolic computation")
    except ImportError:
        logger.warning("SymPy not available - symbolic computation limited")
    
    try:
        import pint
        logger.info("Pint available for unit handling")
    except ImportError:
        logger.warning("Pint not available - using basic unit checking")
    
    logger.info(f"Hands module initialized with {len(ToolRegistry.list_tools())} tools")

_init_module()


# Export physics constants for easy access
from .sympy_solver import SymPySolver
from .unit_checker import UnitChecker

# Make constants available at module level
g = PhysicsConstants.g
c = PhysicsConstants.c
h = PhysicsConstants.h
e = PhysicsConstants.e
kB = PhysicsConstants.kB
EOF
