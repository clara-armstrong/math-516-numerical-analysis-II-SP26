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

        super().__init__(degree, a = 0.0, b = 1.0)

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
            Evaluates B_{index, n}(x).

            Args:
                index (int): basis index (0 <= index <= n)
                x (np.ndarray): evaluation points

            Returns:
                np.ndarray: value of the Bernstein basis polynomial at x

            Raises:
                ValueError: if index is out of range or if x is outside [0, 1]
        """
        if index < 0 or index > self.degree:
            raise ValueError(f"Basis index {index} out of range for degree {self.degree}")
        
        # Ensure x is an array to handle scalar inputs correctly
        x = np.asarray(x)

        if np.any((x < 0) | (x > 1)):
            raise ValueError("Bernstein basis is defined on [0,1]")
        

        return comb(self.degree, index) * (x ** index) * ((1 - x) ** (self.degree - index))
    
    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """

        Solving the linear system Ac = y to force interpolation

    
        Computes coefficients by solving A c = y,
        where A_ij = B_{j,n}(x_i).

        Args:
            x_nodes (np.ndarray): interpolating nodes
            y_nodes (np.ndarray): function values at nodes

        Returns:
            np.ndarray: coefficients of the bernstein polynomial

        Raises:
            ValueError: if system is singular or input sizes mismatch

        """
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)

        if len(x_nodes) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} nodes for degree {self.degree}, got {len(x_nodes)}")


        # Build Bernstein matrix A
        A = np.zeros((self.n_dofs, self.n_dofs))
        for i in range(self.n_dofs):
            for j in range(self.n_dofs):
                A[i, j] = comb(self.degree, j) * (x_nodes[i] ** j) * ((1 - x_nodes[i]) ** (self.degree - j))

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
            coefficients (np.ndarray): coefficients c_i
            x (np.ndarray): evaluation points

        Returns:
            np.ndarray: polynomial value at x

        Raises:
            ValueError: if coefficient size does not match basis dimension
        """
        if len(coefficients) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} coefficients.")

        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        for i in range(self.n_dofs):
            result += coefficients[i] * self.evaluate_basis(i, x)

        return result
   