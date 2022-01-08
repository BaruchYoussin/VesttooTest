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

epochs = 100  # no batches: each epoch is one run
lr_schedule = [1e-3] + 4 * [1e-3] + (epochs - 5) * [1e-2]
model = lib.Arima_0_1_1()
optim = torch.optim.SGD(model.parameters(), lr=lr_schedule[0])
previous_loss = None
for n_iteration, lr in enumerate(lr_schedule):
    optim.param_groups[0]["lr"] = lr
    loss = lib.loss(model, train)
    # Do some early stopping even though all the results are complete overfit:
    if previous_loss is not None and 1 > loss > 0.9 * previous_loss or 0.2 > loss > 0.8 * previous_loss:
        print(f"Loss: {loss:.5f}, early stopping")
        break
    previous_loss = loss.item()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"Iter: {n_iteration}, Loss: {loss:.5f}, c={model.arma_const.item():.5f}, theta={model.ma_coeff().item():.5f}, "
          f"sigma:{model.std_innovation.item():.5f}")
