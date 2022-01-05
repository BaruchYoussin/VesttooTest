# ARIMA(0,1,1) module
import numpy as np
import torch.nn


class Arima_0_1_1(torch.nn.Module):
    """ARIMA(0,1,1) time series.

    Represent the given data, a one time block, as a part of an unlimited ARIMA(0,1,1) time series,
    and learn the optimal ARIMA(0,1,1) parameters.
    The innovation terms are assumed to be normally distributed with zero mean.
    As such representation for given parameter values is not unique, we choose the least squares of the normalized
    innovation values; it is the maximal likelihood choice."""
    def __init__(self, arma_const=None, ma_coeff=None, std_innovation=None):
        super().__init__()
        self.arma_const = torch.nn.Parameter(torch.randn(1, dtype=torch.float) if arma_const is None
                                             else torch.tensor(arma_const, dtype=torch.float))
        self.ma_coeff = torch.nn.Parameter(torch.randn(1, dtype=torch.float) if ma_coeff is None
                                           else torch.tensor(ma_coeff, dtype=torch.float))
        self.std_innovation = torch.nn.Parameter(1 + torch.randn(1, dtype=torch.float) if std_innovation is None
                                                 else torch.tensor(std_innovation, dtype=torch.float))

    def _ma_matrix(self, data_length: int) -> torch.Tensor:
        """Returns the matrix of the moving average transformation from innovations to their moving averages.

        :param data_length: The length of the data block to be generated by moving average model.
        :returns (data_length, data_length + 1) tensor with self.ma_coeff on the main diagonal
            and 1 on the diagonal above it.  See the doc.
        """
        return self.ma_coeff * torch.tensor(
            np.eye(data_length, M=data_length + 1, k=0, dtype=np.float32)) + torch.tensor(
            np.eye(data_length, M=data_length + 1, k=1, dtype=np.float32))

    def _solve_for_ma_innovations(self, data: torch.Tensor) -> torch.Tensor:
        """Solves for innovations producing a given a block of data by MA(1) model.

        :param data: 1-dim tensor.
        :returns The tensor of innovations, 1-dim of length = length (data) = 1, see the doc.
        """
        size = data.size()
        assert len(size) == 1
        length = size[0]
        solution, _, _, _ = torch.linalg.lstsq(self._ma_matrix(length), data - self.arma_const, driver="gels")
        return solution

    def forward(self, time_block: torch.Tensor) -> torch.Tensor:
        """Find the best (the least squares) normalized innovations that produce a given contiguous time block of data.

        :param time_block: a 1-dim object convertible to a 1-dim tensor.
        :returns The tensor of innovations normalized to std = 1.
        """
        if not isinstance(time_block, torch.Tensor):
            time_block = torch.tensor(time_block, dtype=torch.float)
        return self._solve_for_ma_innovations(torch.diff(time_block)) / self.std_innovation


def ma_matrix(ma_coeff: float, data_length: int) -> np.array:
    """Returns the matrix of the moving average transformation from innovations to their moving averages.

    :param ma_coeff: the coefficient of the MA(1) model.
    :param data_length: The length of the data block to be generated by moving average model.
    :returns (data_length, data_length + 1) array with ma_coeff on the main diagonal
        and 1 on the diagonal above it.  See the doc.
    """
    return ma_coeff * np.eye(data_length, M=data_length + 1, k=0) + np.eye(data_length, M=data_length + 1, k=1)


def generate_arima_0_1_1(length: int, arma_const: float, ma_coeff: float, std_innovation:float,
                         initial_value:float) -> np.array:
    innovations = np.random.default_rng().normal(loc=0, scale= std_innovation, size=length)
    moving_averages = np.matmul(ma_matrix(ma_coeff, length - 1), innovations) + arma_const
    return np.cumsum(np.concatenate(([initial_value], moving_averages)))
