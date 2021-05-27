import unittest

import pandas as pd
import numpy as np
from fractions import Fraction

from pgmpy.estimators import NoisyOrMLE
from pgmpy.models.NoisyOrModel import LikelihoodLeaky
from pgmpy.models.NoisyOrModel import LikelihoodNoLeaky
from pgmpy.models.NoisyOrModel import LeakyLL
from pgmpy.models.NoisyOrModel import GradientLeaky
from pgmpy.models import BayesianModel


def Model1():
    """
    C   A --> D
    \  /
     ∨
     B
    Node D uses characters to represent factor
    """
    model = BayesianModel([("A", "B"), ("C", "B"), ("A", "D")])
    data = pd.DataFrame(
        data={"A": [0, 0, 1], "B": [0, 1, 0],
              "C": [1, 1, 0], "D": ["X", "Y", "X"]})
    return model, data


def Model2():
    """
    X1  X2  X3
      \ | /
       ∨∨∨
        Y
    P(X1 = 1) = 0.1   P(X2 = 1) = 0.2   P(X3 = 1) = 0.3
    [θ0, θ1, θ2, θ3] = [0.9, 0.1, 0.2, 0.3]

    The weights t2 are calculated from these parameters using:
    X1 ∈ [0, 1]  X2 ∈ [0, 1]  X3 ∈ [0, 1]  Y ∈ [0, 1]
    w(Y, X1, X2, X3) = P(Y | X1, X2, X3) P(X1, X2, X3) * 10^3
    Where P(Y | X1, X2, X3) =  θ0*∑_(i=1:3) θi^Xi if Y == 1
    else 1 - θ0* ∑_(i=1:3) θi^Xi
    """
    model = BayesianModel([("X1", "Y"), ("X2", "Y"), ("X3", "Y")])
    data = pd.DataFrame(
        {"Y": [0, 1, 0, 1] * 4, "X1": [0, 0, 1, 1] * 4,
        "X2": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
        "X3": [0] * 8  + [1] * 8})
    theta = np.array([Fraction(9, 10),
                        Fraction(1, 10),
                        Fraction(2, 10),
                        Fraction(3, 10)])
    weights = np.array([4536000, 504000, 50400, 590600, 226800, 1033200,
                        2520, 137480, 583200, 1576800, 6480, 233520,
                        29160, 510840, 324, 59676])
    return model, data, theta, weights


def Model3():
    """
    Model 3
    X1 --> Y
    [θ0, θ1] = [0.9, 0.1]
    P(X1 = 1) = 0.5

    The weights t3 are calculated from these parameters using:
    X1 ∈ [0, 1]
    Y ∈ [0, 1]
    w(Y, X1) = P(Y | X1) P(X1) * 10^3
    Where P(Y | X1) =  θ0*θ1^X1 if Y == 1 else 1 - θ0*θ1^X1
    """
    model = BayesianModel([("X1", "Y")])
    data = pd.DataFrame({"Y": [1, 1, 0, 0], "X1": [1, 0, 1, 0]})
    theta = np.array([0.9, 0.1])
    weights = np.array([455, 50, 45, 450])
    return model, data, theta, weights


class TestNoisyOrMLEEstimator(unittest.TestCase):

    def test_state_count(self):
        """
        Test the state_counts method on known data
        """
        model, data = Model1()
        estimator = NoisyOrMLE(model, data)
        self.assertEqual(estimator.state_counts("A").values.tolist(),
                         [[2], [1]])
        self.assertEqual(estimator.state_counts("D").values.tolist(),
                         [[1, 1], [1, 0]])

    def test_likelihood(self):
        """
        Test both leaky and no-leaky likelihood calculation using reationals.
        """
        model, data, theta, weights = Model2()
        # All combinations of 4 binary variables.
        four_binaries = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]]).T

        # Likelihood with leak node
        code_likelihoods = np.apply_along_axis(LikelihoodLeaky,
                                               1,
                                               four_binaries,
                                               theta=theta)
        hand_likelihoods = np.array([Fraction(9, 10), Fraction(1, 10),
                                     Fraction(27, 5000), Fraction(4973, 5000),
                                     Fraction(9, 100), Fraction(91, 100),
                                     Fraction(9, 50), Fraction(41, 50),
                                     Fraction(27, 100), Fraction(73, 100)])

        np.testing.assert_array_equal(code_likelihoods, hand_likelihoods)

        # Likelihood without leak node
        code_likelihoods = np.apply_along_axis(LikelihoodNoLeaky,
                                                  1,
                                                  four_binaries,
                                                  theta=theta[1:])
        hand_likelihoods = np.array([Fraction(1, 1), Fraction(0, 1),
                                        Fraction(3, 500), Fraction(497, 500),
                                        Fraction(1, 10), Fraction(9, 10),
                                        Fraction(1, 5), Fraction(4, 5),
                                        Fraction(3, 10), Fraction(7, 10)])
        np.testing.assert_array_equal(code_likelihoods, hand_likelihoods)

    def test_objective_gradient(self):
        """
        Compare gradient and log likelihood to hand-calculated values from
        model 3.

        Log Likelihood:
        ll_n-or = log(∑_(j:yj=0) w_j (log(θ_0) + ∑_(i=1) x_j,i log(θ_i))
                + ∑_(j:yj=0) w_j log(1 - θ_0 * ∏_(i=1) θ_i ^ (X_j,i)))
        Gradient:
        ∂/∂θ_0 ll_n-or = |d_j ∈ D : y_j = 0|_w * 1/θ_0 - ∑_(j:yj=0) w_j
                        * (∏_(i=1) θ_0^x_j,i)
                        / (1 - θ_0 ∏_(i=1) θ_0^x_j,i)
        ∂/∂θ_i ll_n-or = |d_j ∈ D : y_j = 0 ∧ x_j,i = 1|_w * 1/θ_i
                        - ∑_(j:yj=1 ∧ w_j,i=1) w_j (∏_(i'=/= i) θ_i'^x_j,i')
                        / (1 - θ_0 ∏_(i'=1) θ_i'^x_j,i')
        """
        model, data, theta, weights = Model3()
        # All combinations of 2 binary variables.
        y = np.array([[1], [1], [0], [0]])
        X = np.array([[1], [0], [1], [0]])

        # LeakyLL and Graident of model 3 with known theta
        code_ll = LeakyLL(theta, X, y, weights)
        hand_ll = 313.81
        self.assertAlmostEqual(code_ll, hand_ll, 2)

        code_gradient = GradientLeaky(theta, X, y, weights)
        hand_gradient = np.array([0, 0])
        np.testing.assert_array_almost_equal(code_gradient, hand_gradient, 2)

        # LeakyLL and Graident of model 3 with theta = [0.5, 0.5]
        theta = np.array([0.5, 0.5])
        code_ll = LeakyLL(theta, X, y, weights)
        hand_ll = 539.85
        self.assertAlmostEqual(code_ll, hand_ll, 2)

        code_gradient = GradientLeaky(theta, X, y, weights)
        hand_gradient = np.array([-586.666667, 213.333333])
        np.testing.assert_array_almost_equal(code_gradient, hand_gradient, 2)

        # LeakyLL and Graident of model 3 with theta = [0.1, 0.9]
        theta = np.array([0.1, 0.9])
        code_ll = LeakyLL(theta, X, y, weights)
        hand_ll = 1192.7
        self.assertAlmostEqual(code_ll, hand_ll, 2)

        code_gradient = GradientLeaky(theta, X, y, weights)
        hand_gradient = np.array([-4444.444444, 0])
        np.testing.assert_array_almost_equal(code_gradient, hand_gradient, 2)

        # LeakyLL and Graident of model 3 with theta = [0.001, 0.999]
        theta = np.array([0.001, 0.999])
        code_ll = LeakyLL(theta, X, y, weights)
        hand_ll = 3419.88868
        self.assertAlmostEqual(code_ll, hand_ll, 2)

        code_gradient = GradientLeaky(theta, X, y, weights)
        hand_gradient = np.array([-494494.9504, -44.58959005])
        np.testing.assert_array_almost_equal(code_gradient, hand_gradient, 2)

    def test_optimizer(self):
        """
        Learns Noisy-Or CPDs on models 2 and 3 using their weighted
        distributions, expecting to recover true parameters.
        """
        model, data, theta, weights = Model2()
        estimator=NoisyOrMLE(model, data, weights=weights)
        cpd = estimator.estimate_cpd("Y", leaky=True)
        np.testing.assert_array_almost_equal(cpd.inhibitor_probability,
                                             theta,
                                             2)

        model, data, theta, weights = Model3()
        estimator=NoisyOrMLE(model, data, weights=weights)
        cpd = estimator.estimate_cpd("Y", leaky=True)
        np.testing.assert_array_almost_equal(cpd.inhibitor_probability,
                                             theta,
                                             2)
