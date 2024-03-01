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

    def test_swap(self):
        num_qubits = 4
        mps = MPS(num_qubits)
        mps.ry(math.pi / 3, 0)
        mps.ry(math.pi / 6, 3)
        mps.rx(math.pi / 4, 1)
        mps.rx(math.pi / 5, 2)
        mps.cx(0, 1)
        mps.cx(1, 2)
        mps.cx(2, 3)

        mps.swap(0, 1)
        mps.swap(1, 2)
        mps.cx(3, 2)
        mps.swap(1, 2)
        mps.swap(0, 1)

        bitstrings = [bin(i)[2:].zfill(num_qubits)[::-1] for i in range(2**num_qubits)]
        amplitudes = [mps.amplitude(bs).item() for bs in bitstrings]

        answer = [
            (0.735014795415252 - 2.439383103798491e-18j),
            (1.4719616800160387e-16 - 0.17577607739649173j),
            (-0.09892280773133522 + 2.594357178803248e-17j),
            (-9.813077866773592e-17 - 0.1378832439610859j),
            (1.1622647289044608e-16 - 0.06399183622277295j),
            (-0.015303411614024834 + 4.163336342344337e-17j),
            (2.0816681711721685e-17 - 0.08157796142229075j),
            (0.11370718456745474 + 1.4224732503009818e-16j),
            (-5.637851296924623e-17 - 0.047099057987100464j),
            (0.19694662085644354 - 2.7755575615628914e-17j),
            -0.03694570386915632j,
            (-0.02650628644463044 - 5.204170427930421e-18j),
            (-0.05711310967267997 + 3.9898639947466563e-17j),
            (1.3877787807814457e-17 - 0.23882078405301554j),
            (0.4243609899913534 + 1.5265566588595902e-16j),
            (1.3877787807814457e-17 - 0.30445309680588306j),
        ]

        self.assertTrue(np.allclose(amplitudes, answer))

    def test_ghz1(self):
        num_qubits = 3
        mps = MPS(num_qubits)
        mps.h(0)
        mps.cx(0, 1)
        mps.cx(1, 2)
        amp_000 = mps.amplitude("0" * num_qubits).item()
        amp_111 = mps.amplitude("1" * num_qubits).item()

        self.assertAlmostEqual(amp_000, 1 / math.sqrt(2))
        self.assertAlmostEqual(amp_111, 1 / math.sqrt(2))

    def test_ghz2(self):
        num_qubits = 3
        mps = MPS(num_qubits)
        mps.h(0)
        mps.cx(0, 1)
        mps.swap(0, 1)
        mps.cx(1, 2)
        mps.swap(0, 1)
        amp_000 = mps.amplitude("0" * num_qubits).item()
        amp_111 = mps.amplitude("1" * num_qubits).item()

        self.assertAlmostEqual(amp_000, 1 / math.sqrt(2))
        self.assertAlmostEqual(amp_111, 1 / math.sqrt(2))

    def test_ghz3(self):
        num_qubits = 3
        mps = MPS(num_qubits)
        mps.h(0)
        mps.cx(0, 1)
        mps.cx(0, 2)
        amp_000 = mps.amplitude("0" * num_qubits).item()
        amp_111 = mps.amplitude("1" * num_qubits).item()

        self.assertAlmostEqual(amp_000, 1 / math.sqrt(2))
        self.assertAlmostEqual(amp_111, 1 / math.sqrt(2))

    def test_ghz4(self):
        num_qubits = 3
        mps = MPS(num_qubits)
        mps.x(0)
        mps.h(0)
        mps.cx(0, 1)
        mps.cx(0, 2)
        amp_000 = mps.amplitude("0" * num_qubits).item()
        amp_111 = mps.amplitude("1" * num_qubits).item()

        self.assertAlmostEqual(amp_000, 1 / math.sqrt(2))
        self.assertAlmostEqual(amp_111, -1 / math.sqrt(2))

    def test_ghz5(self):
        num_qubits = 50
        mps = MPS(num_qubits)
        mps.x(0)
        mps.h(0)
        for i in range(num_qubits - 1):
            mps.cx(0, i + 1)
        amp_000 = mps.amplitude("0" * num_qubits).item()
        amp_111 = mps.amplitude("1" * num_qubits).item()

        self.assertAlmostEqual(amp_000, 1 / math.sqrt(2))
        self.assertAlmostEqual(amp_111, -1 / math.sqrt(2))

    def test_ccx(self):
        num_qubits = 3
        mps = MPS(num_qubits)
        mps.ccx(0, 1, 2)
        amp_000 = mps.amplitude("000").item()

        self.assertAlmostEqual(amp_000, 1)

        mps = MPS(num_qubits)
        mps.x(0)
        mps.ccx(0, 1, 2)
        amp_100 = mps.amplitude("100").item()

        self.assertAlmostEqual(amp_100, 1)

        mps = MPS(num_qubits)
        mps.x(1)
        mps.ccx(0, 1, 2)
        amp_010 = mps.amplitude("010").item()

        self.assertAlmostEqual(amp_010, 1)

        mps = MPS(num_qubits)
        mps.x(0)
        mps.x(1)
        mps.ccx(0, 1, 2)
        amp_111 = mps.amplitude("111").item()

        self.assertAlmostEqual(amp_111, 1)

        mps = MPS(num_qubits)
        mps.x(0)
        mps.x(1)
        mps.x(2)
        mps.ccx(0, 1, 2)
        amp_110 = mps.amplitude("110").item()

        self.assertAlmostEqual(amp_110, 1)
