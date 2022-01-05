import lib

data = lib.generate_arima_0_1_1(length=20, arma_const=1, ma_coeff=0.5, std_innovation=1, initial_value=-1)
print(data)
