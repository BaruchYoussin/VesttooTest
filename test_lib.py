import math
from unittest import TestCase
import numpy as np
import torch

import lib


class Test(TestCase):
    def test__solve_for_ma_innovations(self):
        # model = lib.Arima_0_1_1(arma_const=0, ma_coeff=1)
        # data = torch.tensor([0], dtype=torch.float)
        # self.assertTrue(lib._solve_for_ma_innovations(model, data).equal(torch.tensor([0, 0], dtype=torch.float)))
        # data = torch.tensor([1], dtype=torch.float)
        # self.assertTrue((lib._solve_for_ma_innovations(model, data) - torch.tensor([0.5, 0.5], dtype=torch.float))
        #                 .abs().max() < 1e-6)
        # data = torch.tensor([0, 0], dtype=torch.float)
        # self.assertTrue(lib._solve_for_ma_innovations(model, data).equal(torch.tensor([0, 0, 0], dtype=torch.float)))
        # data = torch.tensor([1, 0], dtype=torch.float)
        # self.assertTrue(
        #     (lib._solve_for_ma_innovations(model, data) - torch.tensor([2 / 3, 1 / 3, -1 / 3], dtype=torch.float))
        #     .abs().max() < 1e-6)
        # data = torch.tensor([1, 1], dtype=torch.float)
        # self.assertTrue(
        #     (lib._solve_for_ma_innovations(model, data) - torch.tensor([1 / 3, 2 / 3, 1 / 3], dtype=torch.float))
        #     .abs().max() < 1e-6)

        model = lib.Arima_0_1_1(arma_const=0, ma_coeff=1 / 2)
        data = torch.tensor([0], dtype=torch.float)
        self.assertTrue(lib._solve_for_ma_innovations(model, data).equal(torch.tensor([0, 0], dtype=torch.float)))
        data = torch.tensor([1], dtype=torch.float)
        self.assertTrue((lib._solve_for_ma_innovations(model, data) - torch.tensor([2 / 5, 4 / 5], dtype=torch.float))
                        .abs().max() < 1e-6)
        data = torch.tensor([0, 0], dtype=torch.float)
        self.assertTrue(lib._solve_for_ma_innovations(model, data).equal(torch.tensor([0, 0, 0], dtype=torch.float)))
        data = torch.tensor([1, 0], dtype=torch.float)
        self.assertTrue(
            (lib._solve_for_ma_innovations(model, data) - torch.tensor([10 / 21, 16 / 21, -8 / 21], dtype=torch.float))
            .abs().max() < 1e-6)
        data = torch.tensor([1, 1], dtype=torch.float)
        self.assertTrue(
            (lib._solve_for_ma_innovations(model, data) - torch.tensor([2 / 7, 6 / 7, 4 / 7], dtype=torch.float))
            .abs().max() < 1e-6)

        model = lib.Arima_0_1_1(arma_const=1, ma_coeff=1 / 2)
        data = torch.tensor([1, 0], dtype=torch.float)
        self.assertTrue(
            (lib._solve_for_ma_innovations(model, data) - torch.tensor([4 / 21, -2 / 21, -20 / 21], dtype=torch.float))
            .abs().max() < 1e-6)

    def test_solve_for_innovations(self):
        model = lib.Arima_0_1_1(arma_const=0, ma_coeff=1 / 2, std_innovation=0.5)
        data = torch.tensor([1, 2, 0])
        self.assertTrue((lib.solve_for_innovations(model, data) - torch.tensor([12 / 7, 8 / 7, -32 / 7], dtype=torch.float))
                        .abs().max() < 1e-6)
        model = lib.Arima_0_1_1(arma_const=-1, ma_coeff=1 / 3, std_innovation=0.5)
        data = torch.tensor([1, 1, 2])
        self.assertTrue((lib.solve_for_innovations(model, data) - torch.tensor([24 / 91, 174 / 91, 306 / 91],
                                                                               dtype=torch.float))
                        .abs().max() < 1e-6)

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
