"""
Quadrature class with integration method that performs gauss-legendre or gauss-lobatto.

The only public methods are the integration methods.
"""
import numpy as np
from typing import Callable, Tuple
from scipy.special import roots_jacobi, eval_legendre


class Quadrature:

    def __init__(self, rule: str = 'gauss-legendre', n_points: int = 5):
        """
        Parameters
        ----------
        rule : str
            'gauss-legendre' or 'gauss-lobatto'
        n_points : int
            Number of quadrature points
        """
        if rule != 'gauss-legendre' and rule != 'gauss-lobatto':
            raise ValueError("Unsupported quadrature rule")

        if rule == 'gauss-lobatto' and n_points < 2:
            raise ValueError("Need at least 2 points for Gauss-Lobatto "
                             "quadrature")
        elif n_points < 1:
            raise ValueError("Need at least 1 point for Gauss-Legendre "
                             "quadrature")

        self.rule = rule
        self.n_points = n_points
        self.points: np.ndarray = None
        self.weights: np.ndarray = None

        self._set_quadrature_rule()

    def _set_quadrature_rule(self) -> None:
        """
        Set reference points and weights on [-1, 1].
        """
        if self.rule == 'gauss-lobatto':
            self.points, self.weights = self._gauss_lobatto_points_and_weights()
        else:
            self.points, self.weights = self._gauss_legendre_points_and_weights()

    def _gauss_legendre_points_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Legendre points and weights on [-1, 1].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Points and weights for Gauss-Legendre quadrature.
        """
        return roots_jacobi(self.n_points, 0.0, 0.0)

    def _gauss_lobatto_points_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Lobatto points and weights on [-1, 1].

        To avoid rootfinding algorithms and to minimize function dependencies
        I decided use Scipy's roots_jacobi() function. As it turns out,
        the n-1 degree jacobi polynomial with parameters (1,1) is equivalent
        to the first derivative of the n degree legendre polynomial. The
        function computes its roots for us. As for computing weights,
        we need access to the n degree legendre polynomial and to evaluate it
        at the nodes and that is what Scipy's eval_legendre does for us.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Points and weights for Gauss-Lobatto quadrature.
        """
        if self.n_points == 2:
            return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

        degree = self.n_points - 1

        # Get interior points (zeros of L'_degree)
        # These are roots of P_(degree-1)^(1,1)(x)
        interior_points, _ = roots_jacobi(degree - 1, 1.0, 1.0)
        points = np.concatenate([[-1.0], interior_points, [1.0]])

        # Compute weights using formula: 2/(n(n+1)) * 1/[L_n(x_j)]^2
        n = degree
        L_n = eval_legendre(n, points)
        weights = 2.0 / (n * (n + 1)) * (1.0 / (L_n ** 2))

        return points, weights

    def _validate_bounds(self, a: float, b: float) -> None:
        """
        Raises errors if its not a valid bounded interval [a,b].
        """
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError(f"Integration bounds must be finite. Got a={a}, "
                             f"b={b}")
        if a > b:
            raise ValueError(f"Lower bound a={a} must be <= upper bound b={b}")

    def _affine_map(self, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
        x_phys = (b - a) / 2 * self.points + (a + b) / 2
        weights_phys = (b - a) / 2 * self.weights
        return x_phys, weights_phys

    def integrate(self, f: Callable, a: float, b: float) -> float:
        """
        Computes int_a^b f(x) dx using quadrature.

        Parameters
        ----------
        f : Callable
            The function f(x) to integrate.
        a : float
            Lower bound of integration.
        b : float
            Upper bound of integration.

        Returns
        -------
        float
            Approximate value of the definite integral.
        """
        self._validate_bounds(a, b)
        x_phys, weights_phys = self._affine_map(a, b)
        f_vals = f(x_phys)
        return np.sum(weights_phys * f_vals)

    def integrate_polynomial_object(self, basis, coefficients: np.ndarray,
                                    a: float = None, b: float = None) -> float:
        """
        Convenience method to integrate a polynomial from a basis + coefficients.

        This is just a wrapper around integrate() that creates a callable
        function for you.

        Parameters
        ----------
        basis : object
            Basis object with evaluate() method.
        coefficients : np.ndarray
            Array of shape (n_dofs,) containing basis coefficients.
        a : float, optional
            Lower bound of integration. If None, uses basis.a.
        b : float, optional
            Upper bound of integration. If None, uses basis.b.

        Returns
        -------
        float
            Approximate value of the definite integral.
        """
        if a is None:
            a = basis.a
        if b is None:
            b = basis.b
        self._validate_bounds(a, b)

        f = lambda x: basis.evaluate(coefficients, x)
        return self.integrate(f, a, b)