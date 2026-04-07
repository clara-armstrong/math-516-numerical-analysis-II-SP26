"""
Module's docstring to add
"""
import warnings
import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis

class PowerBasis(PolynomialBasis):
    """
    Implements the standard Monomial Basis: {1, x, x^2, ..., x^n}.
    """
    def __init__(self, degree: int):
        super().__init__(degree, a = 0, b = 1)



    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluates x^index.

            Args:
                index: the index associated to the basis you want to evaluate
                x: The value of x at which to evaluate the polynomial basis.

            Returns:
                The value of the basis at x.
        """
        if index < 0 or index > self.degree:
            raise ValueError(f"Basis index {index} out of range for degree {self.degree}")
        
        # Ensure x is an array to handle scalar inputs correctly
        x = np.asarray(x)
        return np.power(x, index)

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Computes coefficients using the Vandermonde matrix.
        Solves V * c = y, where V_ij = x_i^j.
        """
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)

        if len(x_nodes) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} nodes for degree {self.degree}, got {len(x_nodes)}")

        # np.vander with increasing=True produces columns [x^0, x^1, ..., x^n]
        # This matches our index convention (c_0 corresponding to x^0)
        vander_mat = np.vander(x_nodes, N=self.n_dofs, increasing=True)
        
        # Check condition number (Vandermonde is notoriously ill-conditioned)
        cond_num = np.linalg.cond(vander_mat)
        if cond_num > 1e12:
            warnings.warn(f"Vandermonde matrix is ill-conditioned (cond={cond_num:.2e}). Results may be inaccurate.")

        try:
            coefficients = np.linalg.solve(vander_mat, y_nodes)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered. Ensure nodes are distinct.")
            
        return coefficients

    def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Override: Evaluates a polynomial using Horner's method for numerical stability and speed.

            Args:
                coefficients: A list of the polynomial coefficients, 
                              from lowest to highest power (c0, c1, ..., cn).
                x: The value of x at which to evaluate the polynomial.

            Returns:
                The value of the polynomial at x.
        """
        if len(coefficients) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} coefficients.")
            
        # Start with the highest power coefficient (an)
        result = coefficients[-1]
    
        # Iterate from the second highest coefficient down to c0
        for i in range(len(coefficients) - 2, -1, -1):
            # The core of Horner's method: result = (result * x) + next_coefficient
            result = (result * x) + coefficients[i]
            
        return result
