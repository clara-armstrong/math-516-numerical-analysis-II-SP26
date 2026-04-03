"""
Bernstein polynomial basis

Here we implement the Bernstein polynomial on the reference interval [0, 1]
"""
import warnings
import numpy as np
from math import comb
from numanalysislib.basis._abstract import PolynomialBasis

class BernsteinBasis(PolynomialBasis):
    """  

    Implements the Bernstein Basis: {B_{0,n}, ..., B_{i,n}} on [0,1].

    """
    def __init__(self, degree:int) -> None:

        super().__init__(degree)

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
            Evaluates B_{index, n}(x).

            Args:
                index: basis index (0 <= index <= n)
                x: evaluation points

            Returns:
                Value of the Bernstein basis polynomial at x
        """
        if index < 0 or index > self.degree:
            raise ValueError(f"Basis index {index} out of range for degree {self.degree}")
        
        # Ensure x is an array to handle scalar inputs correctly
        x = np.asarray(x)
        n = self.degree

        if np.any((x < 0) | (x > 1)):
            raise ValueError("Bernstein basis is defined on [0,1]")
        

        return comb(n, index) * (x ** index) * ((1 - x) ** (n - index))
    
    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """

        Solving the linear system Ac = y to force interpolation

    
        Computes coefficients by solving A c = y,
        where A_ij = B_{j,n}(x_i).

        """
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)

        if len(x_nodes) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} nodes for degree {self.degree}, got {len(x_nodes)}")

        n = self.degree

        # Build Bernstein matrix A
        A = np.zeros((self.n_dofs, self.n_dofs))
        for i in range(self.n_dofs):
            for j in range(self.n_dofs):
                A[i, j] = comb(n, j) * (x_nodes[i] ** j) * ((1 - x_nodes[i]) ** (n - j))

        try:
            coefficients = np.linalg.solve(A, y_nodes)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered. Ensure nodes are valid.")
        
        # Check the condition number of A
        cond_num = np.linalg.cond(A)
        if cond_num > 1e12:
            warnings.warn(f"Bernstein matrix is ill-conditioned (cond={cond_num:.2e}).")

        return coefficients


    def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the Bernstein polynomial.

        Args:
            coefficients: coefficients c_i
            x: evaluation points

        Returns:
            Polynomial value at x
        """
        if len(coefficients) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} coefficients.")

        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        for i in range(self.n_dofs):
            result += coefficients[i] * self.evaluate_basis(i, x)

        return result
    