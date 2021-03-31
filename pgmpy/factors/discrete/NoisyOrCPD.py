from itertools import product
from itertools import chain
from warnings import warn
import numbers
import math

import numpy as np
import pandas as pd

from pgmpy.factors.discrete import TabularCPD


class NoisyOrCPD(TabularCPD):
    """
    CPD class for the Noisy-Or model.

    Model is restricted to boolean variables.

    Reference: http://xenon.stanford.edu/~srinivas/research/6-UAI93-Srinivas-Generalization-of-Noisy-Or.pdf
    """

    def __init__(
            self,
            variable,
            variable_card,
            evidence,
            evidence_card,
            inhibitor_probability,
            leaky=False
    ):
        """
        Init method for NoisyOrCPD.

        Parameters
        ----------
        variable: int, string (any hashable python object)
          The variable whose CPD is defined.

        variable_card: integer
          cardinality of variable

        evidence: list, tuple, dict (array like)
          array containing names of the evidence variables.

        evidence_card: list, tuple, dict (array like)
          array containing integers representing the cardinality
          of the evidence variables.

        inhibitor_probability: list, tuple, dict (array_like)
          array containing the inhibitor probabilities of each variable.

        leaky: boolean
          boolean indicating if leaky null term is included in the model.

        Examples
        --------
        >>> from pgmpy.models import NoisyOrModel
        >>> model = NoisyOrModel('D', 2, ['A', 'B', 'C'], [2, 2, 2],
                                                                [0.1,
        ...                                                      0.1,
        ...                                                      0.1])
        """

        if len(evidence) != len(evidence_card):
            raise ValueError(
                            'Size of evidence and'\
                            'evidence_card should be same')
        elif (len(evidence) + leaky) != len(inhibitor_probability):
            raise ValueError(
                'Size of evidence and inhibitor_probability should be same'
            )
        elif not all(
            0 <= item <= 1 for item in inhibitor_probability
        ):
            raise ValueError(
                'Probability values should be between 0 and 1(both inclusive).'
            )
        elif not isinstance(variable_card, int):
            raise ValueError('Variable cardinality should be an integer')
        elif variable_card != 2:
            raise ValueError('Child variable must be binary')
        elif not all (
            item == 2 for item in evidence_card
        ):
            raise ValueError('Parent variables must be binary')
        else:
            inhibitor_index = evidence.copy()
            if leaky:
                inhibitor_index.insert(0, 'leak')
            self.inhibitor_probability = pd.Series(data=inhibitor_probability,
                                                   index=inhibitor_index)
            self.leaky = leaky
            # Blank CPT added -- need to add
            # functionality to convert to CPT later.
            cpt = np.zeros((variable_card, np.product(evidence_card)))
            super(NoisyOrCPD, self).__init__(
                variable,
                variable_card,
                cpt,
                evidence,
                evidence_card
            )
    def __str__(self):
        return str(self.inhibitor_probability)

    def likelihood(self, data):
        """
        Calculates the likelihood of a single data observation given the model.

        data : pandas Series object
            A Series object with axis labels the same as variables in the
            model.
        """

        inhibitor_probability = self.inhibitor_probability
        child = self.variables[0]
        parents = self.variables[1:]
        leaky = self.leaky
        child_data = data[child]
        parent_data = data[parents]

        if leaky:
            # Extract leaky likelihood from inhibitor probabilities
            p0 = inhibitor_probability[0]
            inhibitor_probability = inhibitor_probability[1:]
            def leaky_parent_likelihood(x, pi, p0=p0):
                """Likelihood of single edge in Leaky Noisy-Or"""
                if x == 0:
                    return 1
                else:
                    return (1-pi) / (1-p0)
            # Calculate product of inhibitor probabilities across positive
            # instances in parent_data.
            prod_term = parent_data.combine(inhibitor_probability,
                                            leaky_parent_likelihood)
            if child_data == 0:
                return (1-p0) * np.prod(prod_term)
            else:
                return 1-(1-p0) * np.prod(prod_term)
        else:
            def parent_likelihood(x, pi):
                """Likelihood of single edge in Noisy-Or"""
                if x == 0:
                    return 1
                else:
                    return 1 - pi
            # Calculate product of inhibitor probabilities across positive
            # instances in parent_data.
            prod_term = parent_data.combine(inhibitor_probability,
                                            parent_likelihood)
            if child_data == 0:
                return np.prod(prod_term)
            else:
                return 1 - np.prod(prod_term)

    def get_row_likelihoods(self, data):
        """
        Calculates likelihood of each observation in the data given the model.

        data : pandas DataFrame object
            A DataFrame object with column names same as the variables
            in the model.
        """

        return data.apply(self.likelihood, axis = 1)

    def get_likelihood(self, data):
        """Calculates likelihood of data given the model"""

        return np.prod(self.get_row_likelihoods(data))

    def get_log_likelihood(self, data):
        """Calculates log likelihood of data given the model"""

        ll = self.get_row_likelihoods(data).apply(math.log)
        return np.sum(ll)
