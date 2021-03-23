# coding:utf-8

from itertools import chain

import numpy as np

from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete.NoisyOrCPD import NoisyOrCPD
from pgmpy.models import BayesianModel


class NoisyOrMLE(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Maximum Likelihood Estimation.

        Parameters
        ----------
        model: A pgmpy.models.BayesianModel instance

        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.NaN` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> estimator = NoisyOrMLE(model, data)
        """

        if not isinstance(model, BayesianModel):
            raise NotImplementedError(
                "Noisy Or Estimate is only implemented for BayesianModel"
            )

        super(NoisyOrMLE, self).__init__(model, data, **kwargs)

    def get_parameters(self):
        """
        Method to estimate the model parameters (CPDs) using Maximum Likelihood Estimation.

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
        >>> print(cpd_C.inhibitor_probability)
        [0.6666666666666666, 0.3333333333333333]
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("estimate_cpd: ", node)
        parents = self.model.get_parents(node)
        print("parents: ", parents)

        parent_cardinalities = []
        inhibitor_probability = []
        node_cardinality = len(self.state_names[node])
        # Extract state counts from data
        state_counts = self.state_counts(node)
        state_counts.loc[:, (state_counts == 0).all()] = 1

        print('State Counts:')
        print(state_counts)

        # A Noisy-Or CPD cannot be fit to a node with no parents.
        # Learn a MLE CPT on nodes with no parents.
        if not parents:
            mle = MaximumLikelihoodEstimator(self.model, self.data)
            return mle.estimate_cpd(node)

        # Calculate leak node
        if leaky:
            leak_matrix = state_counts.loc[:, (0,0)]
            leak_parameter = leak_matrix.loc[1] / leak_matrix.sum()
            inhibitor_probability.append(leak_parameter)

        for parent in parents:
            parent_cardinalities.append(len(self.state_names[parent]))
            # Calculate inhibitor probability
            count = state_counts.groupby([parent], axis=1).sum()
            failures = count.iat[0,1]
            trials = count.sum(axis=0).iat[1]
            inhibitor_probability.append(failures / trials)

        print('Node')
        print(node)
        print('Node Cardinality')
        print(node_cardinality)
        print('Parents')
        print(parents)
        print('Parent Cardinalities')
        print(parent_cardinalities)
        print('Inhibitor Probabilities')
        print(inhibitor_probability)

        cpd = NoisyOrCPD(
            node,
            node_cardinality,
            evidence=parents,
            evidence_card=parent_cardinalities,
            inhibitor_probability=np.array(inhibitor_probability),
            leaky=leaky
        )

        return cpd
