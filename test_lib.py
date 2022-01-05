from unittest import TestCase

import torch

import lib


class Test(TestCase):
    def test__solve_for_ma_innovations(self):
        model = lib.Arima_0_1_1(arma_const=0, ma_coeff=1)
        data = torch.tensor([0], dtype=torch.float)
        self.assertTrue(lib._solve_for_ma_innovations(model, data).equal(torch.tensor([0, 0], dtype=torch.float)))
        data = torch.tensor([1], dtype=torch.float)
        self.assertTrue((lib._solve_for_ma_innovations(model, data) - torch.tensor([0.5, 0.5], dtype=torch.float))
                        .abs().max() < 1e-6)
        data = torch.tensor([0, 0], dtype=torch.float)
        self.assertTrue(lib._solve_for_ma_innovations(model, data).equal(torch.tensor([0, 0, 0], dtype=torch.float)))
        data = torch.tensor([1, 0], dtype=torch.float)
        self.assertTrue(
            (lib._solve_for_ma_innovations(model, data) - torch.tensor([2 / 3, 1 / 3, -1 / 3], dtype=torch.float))
            .abs().max() < 1e-6)
        data = torch.tensor([1, 1], dtype=torch.float)
        self.assertTrue(
            (lib._solve_for_ma_innovations(model, data) - torch.tensor([1 / 3, 2 / 3, 1 / 3], dtype=torch.float))
            .abs().max() < 1e-6)

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
        data = [1, 2, 0]
        self.assertTrue((lib.solve_for_innovations(model, data) - torch.tensor([12 / 7, 8 / 7, -32 / 7], dtype=torch.float))
                        .abs().max() < 1e-6)
        model = lib.Arima_0_1_1(arma_const=-1, ma_coeff=1 / 3, std_innovation=0.5)
        data = [1, 1, 2]
        self.assertTrue((lib.solve_for_innovations(model, data) - torch.tensor([24 / 91, 174 / 91, 306 / 91],
                                                                               dtype=torch.float))
                        .abs().max() < 1e-6)
