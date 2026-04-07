"""
**Goal:** Handle the mapping between a "Reference Interval" (e.g., $[-1, 1]$) and a "Physical Interval" $[a, b]$.

- **Methods:** - `pull_back(x)`: Maps $x \in [a, b] \to \hat{x} \in [\hat{a}, \hat{b}]$.
  - `push_forward(hat_x)`: Maps $\hat{x} \in [\hat{a}, \hat{b}] \to x \in [a, b]$.
- **Math:** $x = a + \frac{\hat{x} - \hat{a}}{\hat{b} - \hat{a}}(b - a)$.
- **Note:** All other basis implementations will inherit from this class to gain domain-flexibility.
"""
from numanalysislib.basis._abstract import PolynomialBasis
import numpy as np

class AffinePolynomialBasis(PolynomialBasis):
    def __init__(self, basis: PolynomialBasis, a: float, b: float):
        """
        Initializes class mapping between [a,b] and [hat_a, hat_b]

        Inputs:
        basis: the polynomial basis to use
        a: left endpoint of physical interval
        b: right endpoint of physical interval
        """
        # check for endpoint ordering
        if a > b:
            raise ValueError("b must be greater than a")
        
        # inheret necessary attributes from the basis
        self.basis = basis
        self.a_hat = basis.a
        self.b_hat = basis.b
        self.degree = basis.degree
        self.n_dofs = basis.n_dofs

        # create physical interval
        self.a = a
        self.b = b

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the i-th basis function at points x.
        phi_i(x)
        """
        x_hat = self.pull_back(x)
        x_hat_eval = self.basis.evaluate_basis(index, x_hat)

        return x_hat_eval
    
    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Computes the coefficients c such that p(x_nodes) = y_nodes.
        Returns the coefficients array.
        """
        x_nodes_hat = self.pull_back(x_nodes)
        y_nodes_hat = self.pull_back(y_nodes)
        return self.basis.fit(x_nodes_hat, y_nodes_hat)


    def pull_back(self, x: float) -> float:
        """
        Maps $x \in [a, b] \to \hat{x} \in [\hat{a}, \hat{b}]

        Inputs:
        x: element of physical interval to be mapped into reference interval

        Outputs:
        hat_x: corresponding element in reference interval
        """
        return self.a_hat + (x - self.a)/(self.b - self.a)*(self.b_hat - self.a_hat)

    def push_forward(self, hat_x: float) -> float:
        """
        Maps $\hat{x} \in [\hat{a}, \hat{b}] \to x \in [a, b]$.  

        Inputs:
        hat_x: element of reference interval to be mapped into physical interval

        Outputs:
        x: corresponding element in physical interval
        """
        return self.a + (hat_x - self.a_hat)/(self.b_hat - self.a_hat)*(self.b - self.a)
    