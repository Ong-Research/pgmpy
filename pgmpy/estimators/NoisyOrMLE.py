# coding:utf-8
from itertools import chain
import numpy as np
import pandas as pd
import math

from scipy.optimize import minimize
from scipy.optimize import Bounds

from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete.NoisyOrCPD import NoisyOrCPD
from pgmpy.models import BayesianModel
from pgmpy.models import NoisyOrModel
from pgmpy.models.NoisyOrModel import LeakyLL
from pgmpy.models.NoisyOrModel import NoLeakyLL
from pgmpy.models.NoisyOrModel import GradientLeaky
from pgmpy.models.NoisyOrModel import GradientNoLeaky


class NoisyOrMLE(ParameterEstimator):
    def __init__(self, model, data, weights=None, **kwargs):
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
                "Noisy-Or Estimate is only implemented for BayesianModel"
            )
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Data is not pandas DataFrame"
            )
        if weights is None:
            self.weights = np.ones((data.shape[0],))
        else:
            self.weights = weights
            if len(self.weights.shape) > 1:
                self.weights.shape = (self.weights.shape[0],)
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

    def estimate_cpd(self, node, leaky=True):
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
        >>> data = pd.DataFrame(data={'A': [0, 0, 1],
                                      'B': [0, 1, 0],
                                      'C': [1, 1, 0]})
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
        X = np.array(self.data[parents])
        y = np.array(self.data[node])
        y.shape = (y.shape[0], 1)
        w = self.weights
        if not parents:
            # Learn a MLE CPT on nodes with no parents.
            mle = MaximumLikelihoodEstimator(self.model, self.data)
            return mle.estimate_cpd(node)
        else:
            mle = NoisyOrFit(X, y, w, leaky)

        cpd = NoisyOrCPD(
            node,
            node_cardinality,
            evidence=parents,
            evidence_card=parent_cardinalities,
            inhibitor_probability=mle.x,
            leaky=leaky
        )
        return cpd


def NoisyOrFit(X, y, w, leaky):
    """
    Learn NoisyOr parameters from MLE.
    Parameters
    ----------
    X : numpy array
        Parent data
    y : numpy array
        Child data
    w : numpy array
        observation weights

    Returns
    -------
    theta : numpy array
        learned parameters
    """
    n = X.shape[0]
    m = X.shape[1]

    if leaky:
        m += 1
        objective = LeakyLL
        gradient = GradientLeaky
    else:
        objective = NoLeakyLL
        gradient = GradientNoLeaky

    thetaInit = 0.5 * np.ones((m,))
    bounds = Bounds(lb=0, ub=1)
    fit = minimize(objective,
               x0=thetaInit,
               args=(X, y, w),
               method='SLSQP',
               jac=gradient,
               bounds=bounds,
               options={'iprint': -1})
    return fit
