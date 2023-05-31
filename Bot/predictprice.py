import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Download historical data as dataframe
ticker = "ITC.NS"
df = yf.download(ticker, start="2023-05-24", interval="15m", end="2023-05-27")

# Get the maximum and minimum price
max_price = df['High'].max()
min_price = df['Low'].min()

# Fibonacci Levels considering original trend as upward move
diff = max_price - min_price
level1 = max_price - 0.236 * diff
level2 = max_price - 0.382 * diff
level3 = max_price - 0.618 * diff

print("Level 0%: {:.2f}".format(max_price))
print("Level 23.6%: {:.2f}".format(level1))
print("Level 38.2%: {:.2f}".format(level2))
print("Level 61.8%: {:.2f}".format(level3))
print("Level 100%: {:.2f}".format(min_price))

# # Plotting the Fibonacci retracement levels along with the stock chart
# df['Close'].plot(label=ticker)
# plt.axhline(max_price, linestyle='--', alpha=0.5)
# plt.axhline(level1, linestyle='--', alpha=0.5, color='red')
# plt.axhline(level2, linestyle='--', alpha=0.5, color='blue')
# plt.axhline(level3, linestyle='--', alpha=0.5, color='green')
# plt.axhline(min_price, linestyle='--', alpha=0.5)
# plt.title(f"{ticker} Fibonacci Retracement Levels")
# plt.legend()
# plt.show()
