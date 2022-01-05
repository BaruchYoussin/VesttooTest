import torch.optim

import lib

data = lib.generate_arima_0_1_1(length=20, arma_const=1, ma_coeff=0.5, std_innovation=1, initial_value=-1)
print(data)

train = data[:14]

epochs = 10  # no batches: each epoch is one run
lr_schedule = [1e-3] + 4 * [1e-1] + (epochs - 5) * [0.8]
model = lib.Arima_0_1_1()
optim = torch.optim.SGD(model.parameters(), lr=lr_schedule[0])
previous_loss = None
for lr in lr_schedule:
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
    print(f"Loss: {loss:.5f}, c={model.arma_const.item():.5f}, theta={model.ma_coeff.item():.5f}, "
          f"sigma:{model.std_innovation.item():.5f}")
