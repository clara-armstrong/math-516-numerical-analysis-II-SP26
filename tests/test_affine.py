import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.basis.power import PowerBasis
from numanalysislib.plotting import Plotter

class TestAffinePower:
    def test_vector_pull_back(self):
        """
        Test if the interval [-4, 5] is successfully mapped to [-1, 1]
        """

        # define endpoints
        a = -4
        b = 5

        # make instance
        basis = PowerBasis(10)
        affine = AffinePolynomialBasis(basis, a=a, b=b)

        # verify pull back maps interval
        physical_int = np.linspace(a, b, 100)
        reference_int = np.linspace(0, 1, 100)

        mapped_physical_int = affine.pull_back(physical_int)

        np.testing.assert_allclose(mapped_physical_int, reference_int)

    def test_vector_push_forward(self):
        """
        Test if the interval [-1, 1] is successfully mapped to [-4, 5]
        """
        # define endpoints
        a = -4
        b = 5

        # make instance
        basis = PowerBasis(10)
        affine = AffinePolynomialBasis(basis, a=a, b=b)

        # verify push forward maps interval
        physical_int = np.linspace(a, b, 100)
        reference_int = np.linspace(0, 1, 100)

        mapped_reference_int = affine.push_forward(reference_int)

        np.testing.assert_allclose(mapped_reference_int, physical_int)

    def test_bound_order_test_failure(self):
        """
        Check for failure
        """
        a = -2
        b = -4

        with pytest.raises(ValueError, match = "b must be greater than a"):
            basis = PowerBasis(10)
            affine = AffinePolynomialBasis(basis, a=a, b=b)

class TestAffinePlotter:

    @pytest.fixture
    def affine(self):
        basis = PowerBasis(degree=2)
        return AffinePolynomialBasis(basis, a=2, b=3)

    @patch("matplotlib.pyplot.show")
    def test_affine_plot_basis_smoke(self, mock_show, affine):
        """
        Smoke Test: Does plot_basis run without error?
        We mock plt.show() so no window actually pops up during testing.
        """
        plotter = Plotter()
        try:
            plotter.plot_basis(affine)
        except Exception as e:
            pytest.fail(f"plot_basis raised an exception: {e}")
            
        # Verify show() was called exactly once
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_affine_plot_fit_smoke(self, mock_show, affine):
        """Smoke Test: Does plot_fit run without error?"""
        plotter = Plotter()
        
        # Dummy data
        x = np.array([2.0, 2.5, 3])
        y = x**2
        coeffs = np.array([0, 0, 1])
        
        try:
            plotter.plot_fit(affine, coeffs, x, y, domain=(2, 3))
        except Exception as e:
            pytest.fail(f"plot_fit raised an exception: {e}")
            
        mock_show.assert_called_once()