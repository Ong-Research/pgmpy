# coding:utf-8

from itertools import chain

import numpy as np
import math

from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete.NoisyOrCPD import NoisyOrCPD
from pgmpy.models import BayesianModel

from scipy.optimize import minimize
from scipy.optimize import Bounds


class NoisyOrMLE(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Maximum Likelihood
        Estimation.

        Parameters
        ----------
        model: A pgmpy.models.BayesianModel instance

        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names
            of the network. (If some values in the data are missing the data
            cells should be set to `numpy.NaN`. Note that pandas converts each
            column containing `numpy.NaN`s to dtype `float`.)

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(np.random.randint(low=0,
                                                  high=2,
                                                  size=(1000, 5)),
                                                  columns=['A', 'B', 'C',
                                                           'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'),
                                   ('C', 'D'), ('B', 'E')])
        >>> estimator = NoisyOrMLE(model, data)
        """

        if not isinstance(model, BayesianModel):
            raise NotImplementedError(
                "Noisy Or Estimate is only implemented for BayesianModel"
            )
        super(NoisyOrMLE, self).__init__(model, data, **kwargs)

    def get_parameters(self):
        """
        Method to estimate the model parameters (CPDs) using Maximum
        Likelihood Estimation.

        Returns
        -------
        parameters: list
            List of TabularCPDs, one for each variable of the model

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators.NoisyOrMLE import NoisyOrMLE
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D')])
        >>> estimator = NoisyOrMLE(model, values)
        >>> estimator.get_parameters()
        """

        parameters = []
        for node in sorted(self.model.nodes()):
            cpd = self.estimate_cpd(node)
            parameters.append(cpd)
        return parameters

    def estimate_cpd(self, node, leaky=False):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        leaky: boolean
            Indicates whether the leaky version of noisy-or will be used.
            If true, a leaky parameter is estimated for null instances.

        Returns
        -------
        CPD: NoisyOrCPD

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import NoisyOrMLE
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
        >>> cpd_A = NoisyOrMLE(model, data).estimate_cpd('A')
        >>> print(cpd_A)
        +------+----------+
        | A(0) | 0.666667 |
        +------+----------+
        | A(1) | 0.333333 |
        +------+----------+
        >>> cpd_C = NoisyOrMLE(model, data).estimate_cpd('C')
        >>> print(cpd_C)
        A    0.666667
        B    0.333333
        dtype: float64
        """

        parents = self.model.get_parents(node)
        node_cardinality = len(self.state_names[node])
        parent_cardinalities = [len(self.state_names[parent]) for parent in parents]
        state_counts = self.state_counts(node)
        # Add 1 to all cells with 0 counts to avoid divide by 0 errors.
        state_counts.loc[:, (state_counts == 0).all()] = 1

        # Learn a MLE CPT on nodes with no parents.
        if not parents:
            mle = MaximumLikelihoodEstimator(self.model, self.data)
            return mle.estimate_cpd(node)
        if leaky:
            X = np.array(self.data[parents])
            y = np.array(self.data[node])
            y.shape = (y.shape[0], 1)
            mle = fit_NoisyOr(X, y)

        cpd = NoisyOrCPD(
            node,
            node_cardinality,
            evidence=parents,
            evidence_card=parent_cardinalities,
            inhibitor_probability=mle.x,
            leaky=leaky
        )
        return cpd


def NoisyOrLogProb(D, theta):
    """
    Noisy-Or Log probability of single data observation D.
    """
    theta0 = theta[0]
    thetaI = theta[1:]
    y = D[0]
    X = D[1:]
    if y == 0:
        return math.log(theta0) + sum(X*np.log(thetaI))
    if y == 1:
        return math.log(1 - (theta0 * np.prod(np.power(thetaI, X))))
    else:
        return ValueError

def NoisyOrLL(theta, X, y):
    """
    Noisy-Or Log Likelihood of parent data X and child data y.
    """
    D = np.append(y, X, axis=1)
    log_likelihoods = np.apply_along_axis(NoisyOrLogProb, 1, D, theta=theta)
    return -sum(log_likelihoods)

def NoisyOrGradient(theta, X, y):
    """
    Noisy-Or Gradient of parent data X and child data y.
    """
    grad = np.zeros(theta.shape)
    grad[0] = theta0Gradient(theta, X, y)
    grad[1:] = thetaIGradient(theta, X, y)
    return -grad

def theta0Gradient(theta, X, y):
    """
    Noisy-Or leak node gradient.
    """
    theta0 = theta[0]
    thetaI = theta[1:]
    y0_idx = np.where((y == 0).all(axis=1))
    y1_idx = np.where((y == 1).all(axis=1))
    n_y0 = len(y0_idx[0])
    term1 = n_y0 / theta0
    x_y1 = X[y1_idx]
    try:
        term2 = np.apply_along_axis(theta0Quotient, 1, x_y1,
                                    theta0=theta0, thetaI=thetaI)
        return term1 - sum(term2)
    except ValueError:
        print('Cannot apply_along_axis when any iteration dimensions are 0')
        print(x_y1)

def theta0Quotient(x, theta0, thetaI):
    """
    Quotient term in Noisy-Or leak node gradient.
    """
    numerator = np.prod(np.power(thetaI, x))
    denominator = (1 - theta0 * np.prod(np.power(thetaI, x)))
    return numerator / denominator

def thetaIGradient(theta, X, y):
    """
    Gradient of Noisy-Or inhibitor parameters, excluding leak node parameter.
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
        n_x1_y0 = len(x1_y0_idx[0])
        x1_y1 = X[x1_y1_idx]

        term1 = n_x1_y0 / thetai
        term2 = np.apply_along_axis(thetaIQuotient,
                                    1,
                                    x1_y1,
                                    theta0=theta0,
                                    thetaI=thetaI,
                                    i=i)
        grad.append(term1 - sum(term2))
    return grad

def thetaIQuotient(x, theta0, thetaI, i):
    """
    Quotient term in Noisy-Or gradient.
    """
    x_exclude = np.delete(x, i)
    thetaI_exclude = np.delete(thetaI, i)
    numerator = theta0 * np.prod(np.power(thetaI_exclude, x_exclude))
    denominator = 1 - (theta0 * np.prod(np.power(thetaI, x)))
    return numerator / denominator

def fit_NoisyOr(X, y):
    """
    Learn NoisyOr parameters from MLE.
    Parameters
    ----------
    X : numpy array
        Parent data
    y : numpy array
        Child data

    Returns
    -------
    theta : numpy array
        learned parameters
    """
    n = X.shape[0]
    m = X.shape[1] + 1
    thetaInit = 0.5 * np.ones((m,))
    bounds = Bounds(lb=0.001, ub=0.999)
    fit = minimize(NoisyOrLL,
               x0=thetaInit,
               args=(X, y),
               method='L-BFGS-B',
               jac=NoisyOrGradient,
               bounds=bounds,
               options={'iprint': -1})
    return fit
