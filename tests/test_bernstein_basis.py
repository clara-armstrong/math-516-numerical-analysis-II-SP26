import numpy as np
import pytest
from numanalysislib.basis.bernstein import BernsteinBasis


class TestBernsteinBasis:
     
     def test_initialization(self):
        """Verify degree and DOF setup."""
        basis = BernsteinBasis(degree=3)
        assert basis.degree == 3
        assert basis.n_dofs == 4

     def test_evaluate_basis_endpoints(self):
        """
        Check key Bernstein properties at endpoints:
        B_{0,n}(0) = 1 and B_{n,n}(1) = 1
        """
        basis = BernsteinBasis(degree=3)

        x0 = np.array([0.0])
        x1 = np.array([1.0])

        # B_{0,3}(0) = 1
        assert basis.evaluate_basis(0, x0)[0] == 1.0

        # B_{3,3}(1) = 1
        assert basis.evaluate_basis(3, x1)[0] == 1.0
     def test_evaluate_basis_vector(self):
        """Check vectorized evaluation."""
        basis = BernsteinBasis(degree=2)
        x = np.array([0.0, 0.5, 1.0])

        # B_{1,2}(x) = 2x(1-x)
        y = basis.evaluate_basis(1, x)
        expected = 2 * x * (1 - x)

        np.testing.assert_allclose(y, expected, atol=1e-12)

     def test_partition_of_unity(self):
        """
        Sum of Bernstein basis functions should equal 1.
        """
        basis = BernsteinBasis(degree=4)
        x = np.linspace(0, 1, 50)

        total = np.zeros_like(x)
        for i in range(basis.n_dofs):
            total += basis.evaluate_basis(i, x)

        np.testing.assert_allclose(total, np.ones_like(x), atol=1e-12)

     def test_fit_exact_reconstruction(self):
        """
        Check interpolation correctness.
        Fit a quadratic function and verify reconstruction.
        """
        degree = 2
        basis = BernsteinBasis(degree)

        x_nodes = np.array([0.0, 0.5, 1.0])
        y_nodes = x_nodes**2  # simple function

        coeffs = basis.fit(x_nodes, y_nodes)

        x_test = np.linspace(0, 1, 10)
        y_test = basis.evaluate(coeffs, x_test)

        np.testing.assert_allclose(y_test, x_test**2, atol=1e-12)

     def test_evaluate_full_polynomial(self):
        """
        Test evaluate() directly with known coefficients.
        """
        basis = BernsteinBasis(degree=2)

        # coefficients correspond to control points
        coeffs = np.array([0.0, 1.0, 0.0])

        x = np.array([0.5])
        result = basis.evaluate(coeffs, x)

        # B_{1,2}(0.5) = 2 * 0.5 * 0.5 = 0.5
        assert np.isclose(result[0], 0.5)

     def test_invalid_basis_index(self):
        basis = BernsteinBasis(degree=2)
        x = np.array([0.5])

        with pytest.raises(ValueError):
            basis.evaluate_basis(3, x)

     def test_fit_singularity(self):
        """
        Duplicate nodes should cause failure.
        """
        basis = BernsteinBasis(degree=1)

        x = np.array([0.5, 0.5])  # duplicate
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError):
            basis.fit(x, y)

     def test_endpoint_clustered_nodes_stability(self):
         """
         Check that Bernstein fitting remains numerically reasonable
         even when nodes are clustered near endpoints.
         """

         basis = BernsteinBasis(degree=5)

         # Cluster nodes near endpoints
         x_nodes = np.array([0.0, 0.01, 0.05, 0.95, .99, 1.0])

         # Smooth function
         y_nodes = np.sin(np.pi * x_nodes)

         coeffs = basis.fit(x_nodes, y_nodes)

         x_test = np.linspace(0, 1, 50)
         y_true = np.sin(np.pi * x_test)
         y_pred = basis.evaluate(coeffs, x_test)

         error = np.linalg.norm(y_true - y_pred, ord=np.inf)

         # Not exact — just prevents blow-up
         assert error < 1.0