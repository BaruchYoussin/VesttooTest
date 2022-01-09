import math
from unittest import TestCase
import numpy as np
import torch

import lib


class Test(TestCase):
    def test__cov_matrix(self):
        self.assertTrue(torch.tensor([[1.25, 0.5, 0], [0.5, 1.25, 0.5], [0, 0.5, 1.25]])
                        .equal(lib._cov_matrix(0.5, 1, 3)))
        self.assertTrue(torch.tensor([[5, 2, 0], [2, 5, 2], [0, 2, 5]], dtype=torch.float)
                        .equal(lib._cov_matrix(0.5, 2, 3)))

    def test__squares(self):
        self.assertTrue(
            (lib._squares(torch.tensor([2, 3]), arma_const=torch.tensor(1.), cov_matrix=lib._cov_matrix(1, 1, 2))
             - torch.tensor([2], dtype=torch.float))
            .abs().max() < 1e-6)
        self.assertTrue(
            (lib._squares(torch.tensor([2, 3]), arma_const=torch.tensor(1.), cov_matrix=lib._cov_matrix(-1, 1, 2))
             - torch.tensor([14/3], dtype=torch.float))
            .abs().max() < 1e-6)

    def test_loss(self):
        model = lib.Arima_0_1_1(arma_const=1, ma_coeff=0.5, std_innovation=2)
        self.assertTrue(
            (lib.loss(model=model, time_block=[0, 2, 5])
             - torch.tensor([17/21 + math.log(21)], dtype=torch.float))
            .abs().max() < 1e-6)

    def test_prob_estimate(self):
        model = lib.Arima_0_1_1(arma_const=0, ma_coeff=0, std_innovation=1)
        self.assertEqual(1, lib.prob_estimate(model, np.array([0, 0])))
        self.assertTrue(abs(lib.prob_estimate(model, np.array([0, 5]))) < 1e-4)
        model = lib.Arima_0_1_1(arma_const=0, ma_coeff=0.5, std_innovation=1)
        self.assertEqual(1, lib.prob_estimate(model, np.array([0, 0])))
