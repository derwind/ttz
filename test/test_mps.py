from __future__ import annotations

import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import math

import numpy as np

from ttz.mps import MPS


class TestMPS(unittest.TestCase):
    def test_expectation(self):
        num_qubits = 50
        mps = MPS(num_qubits)
        for i in range(num_qubits):
            theta = (3 / 2) * math.pi / num_qubits * (i + 1)
            mps.ry(theta, i)
        for i in range(num_qubits - 1):
            mps.cx(i, i + 1)
        mps.z(8)
        mps.z(10)
        for i in reversed(range(num_qubits - 1)):
            mps.cx(i, i + 1)
        for i in reversed(range(num_qubits)):
            theta = -(3 / 2) * math.pi / num_qubits * (i + 1)
            mps.ry(theta, i)
        expval = mps.amplitude("0" * num_qubits).real.item()
        self.assertAlmostEqual(expval, 0.29920703698)
