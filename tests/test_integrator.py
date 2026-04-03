# Test file (unchanged except for line length in one test)
import numpy as np
import pytest
from numanalysislib.basis.power import PowerBasis
from src.numanalysislib.calculus.integrator import Quadrature


class TestQuadrature:

    def test_default_initialization(self):
        quad = Quadrature()
        assert quad.rule == 'gauss-legendre'
        assert quad.n_points == 5

    def test_rule_error(self):
        with pytest.raises(ValueError):
            quad = Quadrature(rule='gauss-jacobi')

    def test_lobatto_error(self):
        with pytest.raises(ValueError):
            quad = Quadrature(rule='gauss-lobatto', n_points=1)

    def test_points_error(self):
        with pytest.raises(ValueError):
            quad = Quadrature(rule='gauss-legendre', n_points=-1)

    def test_power_basis(self):
        quad = Quadrature(n_points=3)
        power_basis = PowerBasis(2)
        coeff = np.array([1.0, 1.0, 1.0])
        a = 0.0
        b = 1.0
        actual = 1 + 1/2 + 1/3
        np.testing.assert_allclose(actual,
                                   quad.integrate_polynomial_object(
                                       power_basis, coeff, a, b))

    def test_type1_improper_integral(self):
        quad = Quadrature(rule='gauss-lobatto', n_points=10)
        with pytest.raises(ValueError):
            quad.integrate(lambda x: np.exp(-x), 0, np.inf)

    def test_ab_reversed(self):
        quad = Quadrature(rule='gauss-lobatto', n_points=10)
        with pytest.raises(ValueError):
            quad.integrate(lambda x: np.exp(-x), 1, 0)

    def test_runge(self):
        quad = Quadrature('gauss-legendre', n_points=40)

        # Runge function
        f = lambda x: 1 / (1 + 25*x**2)
        result = quad.integrate(f, -1, 1)
        exact = 2/5 * np.arctan(5)
        np.testing.assert_allclose(result, exact, rtol=1e-6)

    def test_improper_integral_divergent(self):
        quad = Quadrature(rule='gauss-legendre', n_points=1000)
        result = quad.integrate(lambda x: 1/x, 0, 1)
        quad_few = Quadrature(rule='gauss-legendre', n_points=100)
        result_few = quad_few.integrate(lambda x: 1/x, 0, 1)

        # With more points, the approximation should get larger
        assert result > result_few

    def test_improper_integral_singular_but_finite(self):
        quad = Quadrature(rule='gauss-legendre', n_points=1000)

        # This integral is finite despite singularity at x=0
        result = quad.integrate(lambda x: 1/np.sqrt(x), 0, 1)
        exact = 2.0

        # Should be close but not exact due to singularity
        np.testing.assert_allclose(result, exact, rtol=1e-3)