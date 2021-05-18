from itertools import product
from itertools import chain
from warnings import warn
import numbers
import math

import numpy as np
import pandas as pd

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import NoisyOrModel
from pgmpy.models.NoisyOrModel import LikelihoodLeaky
from pgmpy.models.NoisyOrModel import LikelihoodNoLeaky

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
            if leaky is not None:
                self.leaky = leaky
            else:
                print(inhibitor_probability)
                print(evidence)
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

    def __repr__(self):
        var_str = f"<NoisyOrCPD representing P({self.variable}:{self.variable_card}"

        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            evidence_str = " | " + ", ".join(
                [f"{var}:{card}" for var, card in zip(evidence, evidence_card)]
            )
        else:
            evidence_str = ""

        return var_str + evidence_str + f") at {hex(id(self))}>"

    def data_likelihoods(self, data):
        """
        Calculates the likelihoods of each observation given the model.

        data : pandas Series object
            A Series object with axis labels the same as variables in the
            model.
        """
        child = self.variables[0]
        parents = self.variables[1:]
        y = np.array(data[child])
        X = np.array(data[parents])
        y.shape = (y.shape[0], 1)
        D = np.append(y, X, axis=1)
        if self.leaky:
            l = LikelihoodLeaky
        else:
            l = LikelihoodNoLeaky
        likelihoods = np.apply_along_axis(l, 1,
                                          D, theta=self.inhibitor_probability)
        return likelihoods

    def likelihood(self, data):
        """
        Calculates the likelihood of the data given the model.

        data : pandas Series object
            A Series object with axis labels the same as variables in the
            model.
        """
        likelihoods = self.data_likelihoods(data)
        return np.prod(likelihoods)

    def log_likelihood(self, data):
        """
        Calculates the log-likelihood of the data given the model.

        data : pandas Series object
            A Series object with axis labels the same as variables in the
            model.
        """
        likelihoods = self.data_likelihoods(data)
        return np.sum(np.log(likelihoods))
