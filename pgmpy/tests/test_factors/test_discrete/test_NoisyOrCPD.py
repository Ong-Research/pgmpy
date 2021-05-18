import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_test
from pgmpy.factors.discrete.NoisyOrCPD import NoisyOrCPD

def Model1():
    return NoisyOrCPD("x1", 2, ["x2", "x3", "x4"], [2, 2, 2], [0.4, 0.7, 0.1], leaky=False)

class TestNoisyOrCPDInit(unittest.TestCase):

    def test_init(self):
        model = Model1()
        np_test.assert_array_equal(model.variables,
                                   np.array(["x1", "x2", "x3", "x4"]))
        np_test.assert_array_equal(model.cardinality, np.array([2, 2, 2, 2]))
        pd.testing.assert_series_equal(model.inhibitor_probability,
                                       pd.Series(data=[0.4, 0.7, 0.1],
                                       index=["x2", "x3", "x4"]))

    def test_exceptions(self):
        self.assertRaises(
            ValueError,
            NoisyOrCPD,
            variable = "x1",
            variable_card = 2,
            evidence = ["x2", "x3"],
            evidence_card = [2, 3],
            inhibitor_probability = [0.4, 0.7],
        )
        self.assertRaises(
            ValueError,
            NoisyOrCPD,
            variable = "x1",
            variable_card = 2,
            evidence = ["x2", "x3"],
            evidence_card = [2, 2],
            inhibitor_probability = [0.4, 0.7, 0.1],
        )
        self.assertRaises(
            ValueError,
            NoisyOrCPD,
            variable = "x1",
            variable_card = 2,
            evidence = ["x2", "x3"],
            evidence_card = [2, 2],
            inhibitor_probability = [0.4, 1.2],
        )
        self.assertRaises(
            ValueError,
            NoisyOrCPD,
            variable = "x1",
            variable_card = 2,
            evidence = ["x2", "x3"],
            evidence_card = [2, 2, 2],
            inhibitor_probability = [0.4, 1.2],
        )
        self.assertRaises(
            ValueError,
            NoisyOrCPD,
            variable = "x1",
            variable_card = 2,
            evidence = ["x2", "x3"],
            evidence_card = [2, 2.1, 2],
            inhibitor_probability = [0.4, 1.2],
        )
        self.assertRaises(
            ValueError,
            NoisyOrCPD,
            variable = "x1",
            variable_card = 3,
            evidence = ["x2", "x3"],
            evidence_card = [2, 2, 2],
            inhibitor_probability = [0.4, 1.2],
        )

    def test_likelihood(self):
        model = Model1()
        data = pd.DataFrame({'x1': [1, 0],'x2': [0, 1], 'x3': [1, 1], 'x4': [0, 1]})
        model.likelihood(data)
