from itertools import product
from itertools import chain
from warnings import warn
import numbers

import numpy as np

from pgmpy.factors.discrete import TabularCPD


class NoisyOrCPD(TabularCPD):
  """
  Base class for Noisy-Or models.

  This is an implementation of generalized Noisy-Or models and
  is not limited to Boolean variables and also any arbitrary
  function can be used instead of the boolean OR function.

  Reference: http://xenon.stanford.edu/~srinivas/research/6-UAI93-Srinivas-Generalization-of-Noisy-Or.pdf
  """

  def __init__(
        self,
        variable,
        variable_card,
        evidence,
        evidence_card,
        inhibitor_probability,
    ):
    # TODO: Accept values of each state so that it could be
    # put into F to compute the final state values of the output
    """
    Init method for NoisyOrModel.

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

    Examples
    --------
    >>> from pgmpy.models import NoisyOrModel
    >>> model = NoisyOrModel('x0', 2, ['x1', 'x2', 'x3'], [2, 2, 2], [[0.6, 0.4],
    ...                                                      [0.4, 0.7],
    ...                                                      [0.1, 0.4]])
    """
    self.inhibitor_probability = []
    if len(evidence) != len(evidence_card):
        raise ValueError("Size of evidence and evidence_ccard should be same")
    elif any(
        cardinal != len(prob_array)
        for prob_array, cardinal in zip(inhibitor_probability, evidence_card)
    ) or len(evidence_card) != len(inhibitor_probability):
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
        blank_cpd = np.ones((variable_card, np.product(evidence_card)))
        super(NoisyOrCPD, self).__init__(variable,
                                            variable_card,
                                            blank_cpd,
                                            evidence,
                                            evidence_card
                                        )
        self.inhibitor_probability.extend(inhibitor_probability)
