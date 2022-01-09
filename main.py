import math

import torch.optim

import lib

data_length = 20
train_length = 14
np_seed = 42
torch_seed = -0x3a9357fb9

data = lib.generate_arima_0_1_1(length=data_length, arma_const=1, ma_coeff=0.5, std_innovation=1, initial_value=-1,
                                seed=np_seed)
print(data)

torch.manual_seed(torch_seed)

train = data[:train_length]

epochs = 10000  # no batches: each epoch is one iteration
lr_schedule = [1e-3] + 4 * [1e-3] + (epochs - 5) * [1e-2]
# Learning usually stabilizes after a few hundred iterations, with no signs of overfit; for these particular seeds,
# it keeps improving slowly but significantly over the first few thousands.
model = lib.Arima_0_1_1()
optim = torch.optim.SGD(model.parameters(), lr=lr_schedule[0])
previous_loss = None
for n_iteration, lr in enumerate(lr_schedule):
    optim.param_groups[0]["lr"] = lr
    loss = lib.loss(model, train)
    previous_loss = loss.item()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"Iter: {n_iteration}, Loss: {loss:.5f}, c={model.arma_const.item():.5f}, theta={model.ma_coeff().item():.5f}, "
          f"sigma:{model.std_innovation.item():.5f}")

test = data[(train_length - 1):]
loss_test = lib.loss(model, test)
prob_density = (2 * math.pi) ** (-(data_length - train_length + 1) / 2) * math.exp(-loss_test / 2)
prob_estimate = lib.prob_estimate(model, test)
print(f"For the test: loss = {loss_test:.5f}, Prob density = {prob_density:.5g}, Prob estimate = {prob_estimate:.5g}")
