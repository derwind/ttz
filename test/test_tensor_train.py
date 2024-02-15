from __future__ import annotations

import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np
import torch
from torch import nn

from ttz.tt import TTLayer


class TestTT(unittest.TestCase):
    def test_ttlayer1(self):
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        layer = nn.Linear(1024, 100)
        ttlayer = TTLayer.from_linear_layer([2**5, 2**5], [10, 10], layer)
        input = torch.randn(1, 1024)
        output1 = layer(input).detach().cpu().numpy()
        output2 = ttlayer(input).detach().cpu().numpy()
        self.assertTrue(np.allclose(output1, output2, atol=1e-6))

    def test_ttlayer2(self):
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        layer = nn.Linear(4096, 216)
        ttlayer = TTLayer.from_linear_layer([2**4, 2**4, 2**4], [6, 6, 6], layer)
        input = torch.randn(1, 4096)
        output1 = layer(input).detach().cpu().numpy()
        output2 = ttlayer(input).detach().cpu().numpy()
        self.assertTrue(np.allclose(output1, output2, atol=1e-5))

    def test_low_rank_approximation(self):
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        layer = nn.Linear(1024, 100)
        ttlayer = TTLayer.from_linear_layer([2**5, 2**5], [10, 10], layer, bond_dims=[10, 99, 32])
        input = torch.randn(1, 1024)
        output1 = layer(input).detach().cpu().numpy()
        output2 = ttlayer(input).detach().cpu().numpy()
        self.assertTrue(np.allclose(output1, output2, atol=8e-2))
