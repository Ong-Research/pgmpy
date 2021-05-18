#!/usr/bin/env python3
from itertools import chain

import numpy as np
import networkx as nx


class NoisyOrModel(nx.DiGraph):
    """
    Base class for Noisy-Or models.

    This is an implementation of generalized Noisy-Or models and
    is not limited to Boolean variables and also any arbitrary
    function can be used instead of the boolean OR function.

    Reference: http://xenon.stanford.edu/~srinivas/research/6-UAI93-Srinivas-Generalization-of-Noisy-Or.pdf
    """

    def __init__(self, variables, cardinality, inhibitor_probability):
        # TODO: Accept values of each state so that it could be
        # put into F to compute the final state values of the output
        """
        Init method for NoisyOrModel.

        Parameters
        ----------
        variables: list, tuple, dict (array like)
            array containing names of the variables.

        cardinality: list, tuple, dict (array like)
            array containing integers representing the cardinality
            of the variables.

        inhibitor_probability: list, tuple, dict (array_like)
            array containing the inhibitor probabilities of each variable.

        Examples
        --------
        >>> from pgmpy.models import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        ...                                                      [0.2, 0.4, 0.7],
        ...                                                      [0.1, 0.4]])
        """
        self.variables = np.array([])
        self.cardinality = np.array([], dtype=np.int)
        self.inhibitor_probability = []
        self.add_variables(variables, cardinality, inhibitor_probability)

    def add_variables(self, variables, cardinality, inhibitor_probability):
        """
        Adds variables to the NoisyOrModel.

        Parameters
        ----------
        variables: list, tuple, dict (array like)
            array containing names of the variables that are to be added.

        cardinality: list, tuple, dict (array like)
            array containing integers representing the cardinality
            of the variables.

        inhibitor_probability: list, tuple, dict (array_like)
            array containing the inhibitor probabilities corresponding to each variable.

        Examples
        --------
        >>> from pgmpy.models import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        ...                                                      [0.2, 0.4, 0.7],
        ...                                                      [0.1, 0., 4]])
        >>> model.add_variables(['x4'], [3], [0.1, 0.4, 0.2])
        """
        if len(variables) == 1:
            if not isinstance(inhibitor_probability[0], (list, tuple)):
                inhibitor_probability = [inhibitor_probability]

        if len(variables) != len(cardinality):
            raise ValueError("Size of variables and cardinality should be same")
        elif any(
            cardinal != len(prob_array)
            for prob_array, cardinal in zip(inhibitor_probability, cardinality)
        ) or len(cardinality) != len(inhibitor_probability):
            raise ValueError(
                "Size of variables and inhibitor_probability should be same"
            )
        elif not all(
            0 <= item <= 1 for item in chain.from_iterable(inhibitor_probability)
        ):
            raise ValueError(
                "Probability values should be between 0 and 1(both inclusive)."
            )
        else:
            self.variables = np.concatenate((self.variables, variables))
            self.cardinality = np.concatenate((self.cardinality, cardinality))
            self.inhibitor_probability.extend(inhibitor_probability)

    def del_variables(self, variables):
        """
        Deletes variables from the NoisyOrModel.

        Parameters
        ----------
        variables: list, tuple, dict (array like)
            list of variables to be deleted.

        Examples
        --------
        >>> from pgmpy.models import NoisyOrModel
        >>> model = NoisyOrModel(['x1', 'x2', 'x3'], [2, 3, 2], [[0.6, 0.4],
        ...                                                      [0.2, 0.4, 0.7],
        ...                                                      [0.1, 0., 4]])
        >>> model.del_variables(['x1'])
        """
        variables = [variables] if isinstance(variables, str) else set(variables)
        indices = [
            index
            for index, variable in enumerate(self.variables)
            if variable in variables
        ]
        self.variables = np.delete(self.variables, indices, 0)
        self.cardinality = np.delete(self.cardinality, indices, 0)
        self.inhibitor_probability = [
            prob_array
            for index, prob_array in enumerate(self.inhibitor_probability)
            if index not in indices
        ]


def LeakyLL(theta, X, y, w):
    """
    Log likelihood of parent data X and child data y with leak node.
    """
    D = np.append(y, X, axis=1)
    likelihoods = np.apply_along_axis(LikelihoodLeaky, 1, D, theta=theta)
    log_likelihoods = np.log(likelihoods)
    weighted_log_likelihoods = np.multiply(w, log_likelihoods)
    return -np.sum(weighted_log_likelihoods)


def NoLeakyLL(theta, X, y, w):
    """
    Log likelihood of parent data X and child data y without
    leak node.
    """
    D = np.append(y, X, axis=1)
    likelihoods = np.apply_along_axis(LikelihoodNoLeaky, 1, D, theta=theta)
    log_likelihoods = np.log(likelihoods)
    weighted_log_likelihoods = np.multiply(w, log_likelihoods)
    return -np.sum(weighted_log_likelihoods)


def LikelihoodLeaky(D, theta):
    """
    Likelihood of single data observation D with leak node.
    y is assumed to be first element of vector D.
    """
    theta0 = theta[0]
    thetaI = theta[1:]
    y = D[0]
    X = D[1:]
    p = theta0 * np.prod(np.power(thetaI, X))
    if y == 0:
        return p
    if y == 1:
        return 1 - p
    else:
        return ValueError


def LikelihoodNoLeaky(D, theta):
    """
    Likelihood of single data observation D without leak node.
    y is assumed to be first element of vector D.
    """
    y = D[0]
    X = D[1:]
    p = np.prod(np.power(theta, X))
    if y == 0:
        return p
    if y == 1:
        return 1 - p
    else:
        return ValueError


def GradientLeaky(theta, X, y, w):
    """
    Leaky gradient of parent data X and child data y.
    """
    grad = np.zeros(theta.shape)
    grad[0] = LeakGradient(theta, X, y, w)
    grad[1:] = InhibitorGradientLeaky(theta, X, y, w)
    return -grad


def GradientNoLeaky(theta, X, y, w):
    """
    No-leaky gradient of parent data X and child data y.
    """
    grad = np.zeros(theta.shape)
    grad[:] = InhibitorGradientNoLeaky(theta, X, y, w)
    return -grad


def LeakGradient(theta, X, y, w):
    """
    Leak node gradient.
    """
    theta0 = theta[0]
    thetaI = theta[1:]
    y0_idx = np.where((y == 0).all(axis=1))
    y1_idx = np.where((y == 1).all(axis=1))
    n_y0 = np.sum(w[y0_idx])
    term1 = n_y0 / theta0
    X_y1 = X[y1_idx]
    w_y1 = w[y1_idx]
    try:
        term2 = np.apply_along_axis(LeakQuotient, 1, X_y1,
                                    theta0=theta0, thetaI=thetaI)
        w_y1.shape = (w_y1.shape[0],)
        term2 = sum(np.multiply(w_y1, term2))
        return term1 - term2
    except ValueError:
        print('0 instances of y == 1 in data')
        print(x_y1)


def LeakQuotient(x, theta0, thetaI):
    """
    Quotient term in leak node gradient.
    """
    numerator = np.prod(np.power(thetaI, x))
    denominator = 1 - theta0 * np.prod(np.power(thetaI, x))
    return numerator / denominator


def InhibitorGradientLeaky(theta, X, y, w):
    """
    Leaky gradient of inhibitor parameters,
    excluding leak node parameter.
    """
    theta0 = theta[0]
    thetaI = theta[1:]
    grad = []
    for i in range(X.shape[1]):
        thetai = thetaI[i]
        x = X[:, i]
        x = np.reshape(x, (x.shape[0], 1))
        D = np.append(x, y, axis=1)
        # Subset data by x=1,y=0 & x=1,y=1
        x1_y0_idx = np.where(np.all(D == (1, 0), axis=1))
        x1_y1_idx = np.where(np.all(D == (1, 1), axis=1))
        n_x1_y0 = np.sum(w[x1_y0_idx])
        x1_y1 = X[x1_y1_idx]
        w_x1_y1 = w[x1_y1_idx]
        term1 = n_x1_y0 / thetai
        term2 = np.apply_along_axis(InhibitorQuotientLeaky,
                                    1,
                                    x1_y1,
                                    theta0=theta0,
                                    thetaI=thetaI,
                                    i=i)
        w_x1_y1.shape = (w_x1_y1.shape[0],)
        term2 = sum(np.multiply(w_x1_y1, term2))
        grad.append(term1 - term2)
    return grad


def InhibitorGradientNoLeaky(theta, X, y, w):
    """
    No-leaky gradient of inhibitor parameters.
    """
    grad = []
    for i in range(X.shape[1]):
        thetai = theta[i]
        x = X[:, i]
        x = np.reshape(x, (x.shape[0], 1))
        D = np.append(x, y, axis=1)
        # Subset data by x=1,y=0 & x=1,y=1
        x1_y0_idx = np.where(np.all(D == (1, 0), axis=1))
        x1_y1_idx = np.where(np.all(D == (1, 1), axis=1))
        n_x1_y0 = np.sum(w[x1_y0_idx])
        x1_y1 = X[x1_y1_idx]
        w_x1_y1 = w[x1_y1_idx]
        term1 = n_x1_y0 / thetai
        term2 = np.apply_along_axis(InhibitorQuotientNoLeaky,
                                    1,
                                    x1_y1,
                                    theta=theta,
                                    i=i)
        term2 = sum(np.multiply(w_x1_y1, term2))
        grad.append(term1 - term2)
    return grad


def InhibitorQuotientLeaky(x, theta0, thetaI, i):
    """
    Leaky quotient term in inhibitor gradient.
    """
    x_exclude = np.delete(x, i)
    thetaI_exclude = np.delete(thetaI, i)
    numerator = theta0 * np.prod(np.power(thetaI_exclude, x_exclude))
    denominator = 1 - (theta0 * np.prod(np.power(thetaI, x)))
    return numerator / denominator


def InhibitorQuotientNoLeaky(x, theta, i):
    """
    No-leaky quotient term in inhibitor gradient.
    """
    x_exclude = np.delete(x, i)
    theta_exclude = np.delete(theta, i)
    numerator = np.prod(np.power(theta_exclude, x_exclude))
    denominator = 1 - np.prod(np.power(theta, x))
    return numerator / denominator
