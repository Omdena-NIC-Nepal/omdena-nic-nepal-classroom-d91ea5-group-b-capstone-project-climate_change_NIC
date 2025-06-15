import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("../Weather & Climate Data/Processed Datas/grouped_5yr.csv")
# Generate sample time series data
np.random.seed(0)
n = 100
time_series = pd.Series(np.random.randn(n).cumsum())
print(time_series.head())
# Plot the time series
# plt.plot(time_series)
# plt.title("Sample Time Series Data")
# plt.show()