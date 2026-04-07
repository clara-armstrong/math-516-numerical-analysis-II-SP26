from abc import ABC, abstractmethod
import numpy as np

class PolynomialBasis(ABC):
    """
    Abstract base class for a generic finite-dimensional vector space 
    of polynomials (or functions).
    """

    def __init__(self, degree: int, a: float, b: float):
        self.degree = degree
        self.n_dofs = degree + 1
        self.a = a
        self.b = b

    @abstractmethod
    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the i-th basis function at points x.
        phi_i(x)
        """
        pass

    def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the polynomial p(x) = sum(c_i * phi_i(x)).
        Default implementation can be overridden for efficiency (e.g., Horner's method).
        """
        if len(coefficients) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} coefficients, got {len(coefficients)}")
        
        y = np.zeros_like(x, dtype=float)
        for i, c in enumerate(coefficients):
            y += c * self.evaluate_basis(i, x)
        return y

    @abstractmethod
    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Computes the coefficients c such that p(x_nodes) = y_nodes.
        Returns the coefficients array.
        """
        pass