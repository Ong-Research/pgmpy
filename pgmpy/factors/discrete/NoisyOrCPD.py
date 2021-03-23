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
        leaky=False
    ):
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

    leaky: boolean
      boolean indicating if leaky null term is included in inhibitor
      probabilities.

    Examples
    --------
    >>> from pgmpy.models import NoisyOrModel
    >>> model = NoisyOrModel('D', 2, ['A', 'B', 'C'], [2, 2, 2],
                                                            [0.1,
    ...                                                      0.1,
    ...                                                      0.1])
    """
    self.inhibitor_probability = []
    if len(evidence) != len(evidence_card):
        raise ValueError("Size of evidence and evidence_card should be same")
    elif (variable_card + leaky) != len(inhibitor_probability):
        raise ValueError(
            "Size of variables and inhibitor_probability should be same"
        )
    elif not all(
        0 <= item <= 1 for item in inhibitor_probability
    ):
        raise ValueError(
            "Probability values should be between 0 and 1(both inclusive)."
        )
    elif not isinstance(variable_card, int):
        raise ValueError("Variable cardinality should be an integer")
    else:
        # Blank CPT added -- need to add
        # functionality to convert to CPT later.
        #if inhibitor_probability is not None:
        self.inhibitor_probability.extend(inhibitor_probability)
        # cpt = get_cpt(variable,
        #     variable_card,
        #     evidence,
        #     evidence_card,
        #     self.inhibitor_probability)
        # else:
        cpt = np.zeros((variable_card, np.product(evidence_card)))

        super(NoisyOrCPD, self).__init__(
            variable,
            variable_card,
            cpt,
            evidence,
            evidence_card
        )
