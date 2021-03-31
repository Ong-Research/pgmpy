import unittest

import pandas as pd
import numpy as np

from pgmpy.estimators import NoisyOrMLE
from pgmpy.models import BayesianModel


class TestNoisyOrMLEEstimator(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "X"]}
        )
        self.d2 = pd.DataFrame(
            data={
                "A": [0, np.NaN, 1],
                "B": [0, 1, 0],
                "C": [1, 1, np.NaN],
                "D": [np.NaN, "Y", np.NaN],
            }
        )
        self.s1 = BayesianModel([("A", "B"), ("C", "B"), ("A", "D")])

    def test_state_count(self):
        e = NoisyOrMLE(self.s1, self.d1)
        self.assertEqual(e.state_counts("A").values.tolist(), [[2], [1]])
        self.assertEqual(
            e.state_counts("D").values.tolist(),
            [[1.0, 1.0], [1.0, 0.0]],
        )

    def test_exceptions(self):
        self.assertRaises(
            NotImplementedError,
            NoisyOrMLE,
            model = [("A", "B"), ("C", "B"), ("A", "D")],
            data = self.d1
        )

    def tearDown(self):
        del self.d1
        del self.d2
