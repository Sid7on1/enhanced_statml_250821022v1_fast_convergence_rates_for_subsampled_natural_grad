# objective_functions.py

import logging
import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    "objective_functions": {
        "quadratic": {
            "coefficients": [1.0, 2.0, 3.0],
            "bounds": [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
        },
        "rosenbrock": {
            "bounds": [(-10.0, 10.0), (-10.0, 10.0)],
        },
    },
}

class ObjectiveFunction:
    def __init__(self, name: str, coefficients: List[float], bounds: List[Tuple[float, float]]):
        self.name = name
        self.coefficients = coefficients
        self.bounds = bounds

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

class QuadraticObjectiveFunction(ObjectiveFunction):
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([coeff * x**2 for coeff in self.coefficients])

class RosenbrockObjectiveFunction(ObjectiveFunction):
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2])

class ObjectiveFunctions:
    def __init__(self):
        self.objective_functions = {
            "quadratic": QuadraticObjectiveFunction(
                name="quadratic",
                coefficients=CONFIG["objective_functions"]["quadratic"]["coefficients"],
                bounds=CONFIG["objective_functions"]["quadratic"]["bounds"],
            ),
            "rosenbrock": RosenbrockObjectiveFunction(
                name="rosenbrock",
                bounds=CONFIG["objective_functions"]["rosenbrock"]["bounds"],
            ),
        }

    def get_objective_function(self, name: str) -> ObjectiveFunction:
        return self.objective_functions.get(name)

    def evaluate_objective_function(self, name: str, x: np.ndarray) -> float:
        objective_function = self.get_objective_function(name)
        if objective_function is None:
            raise ValueError(f"Unknown objective function: {name}")
        return objective_function.evaluate(x)

def create_objective_function(name: str) -> ObjectiveFunction:
    return ObjectiveFunctions().get_objective_function(name)

def evaluate_objective_function(name: str, x: np.ndarray) -> float:
    return ObjectiveFunctions().evaluate_objective_function(name, x)

# Unit tests
if __name__ == "__main__":
    objective_functions = ObjectiveFunctions()

    # Test quadratic objective function
    quadratic_objective_function = objective_functions.get_objective_function("quadratic")
    x = np.array([1.0, 2.0, 3.0])
    result = quadratic_objective_function.evaluate(x)
    logger.info(f"Quadratic objective function result: {result}")

    # Test Rosenbrock objective function
    rosenbrock_objective_function = objective_functions.get_objective_function("rosenbrock")
    x = np.array([1.0, 2.0])
    result = rosenbrock_objective_function.evaluate(x)
    logger.info(f"Rosenbrock objective function result: {result}")