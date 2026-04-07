import pytest
import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis

# -------------------------------------------------------------------------
# 1. Define a Mock Class
# We create a simple "Monomial" basis just for testing the base logic.
# -------------------------------------------------------------------------
class MockBasis(PolynomialBasis):
    def __init__(self, degree: int):
        super().__init__(degree = degree, a = 0, b = 5)
    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        # Simple monomial basis: x^index
        return x ** index

    def fit(self, x_nodes, y_nodes):
        # We don't need to test fit() here, but the ABC requires it exist
        pass

# -------------------------------------------------------------------------
# 2. Test the Concrete Logic Inherited from the ABC
# -------------------------------------------------------------------------
def test_evaluate_summation_logic():
    """
    Test if the base class correctly sums c_i * phi_i(x).
    We use the MockBasis where phi_i(x) = x^i.
    """
    degree = 2
    basis = MockBasis(degree=degree)
    
    # Polynomial: P(x) = 1*x^0 + 2*x^1 + 3*x^2
    coeffs = np.array([1.0, 2.0, 3.0])
    x_points = np.array([0.0, 1.0, 2.0])
    
    # Expected results:
    # x=0: 1 + 0 + 0 = 1
    # x=1: 1 + 2 + 3 = 6
    # x=2: 1 + 4 + 12 = 17
    expected = np.array([1.0, 6.0, 17.0])
    
    result = basis.evaluate(coeffs, x_points)
    
    # Use np.testing for safe floating point comparisons
    np.testing.assert_allclose(result, expected, rtol=1e-14)

def test_coefficient_shape_mismatch():
    """
    The base class should raise an error if we provide the 
    wrong number of coefficients.
    """
    basis = MockBasis(degree=2) # Expects 3 coeffs (0, 1, 2)
    wrong_coeffs = np.array([1.0, 1.0]) # Only 2 provided
    
    x = np.array([0.0])
    
    with pytest.raises(ValueError):
        basis.evaluate(wrong_coeffs, x)