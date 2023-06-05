import pandas as pd
from datetime import datetime
from utility import *

year_month = datetime.today().strftime('%Y')
index, bucket = get_scan_type()

# file paths
tradefile=f'Trades/trades_{year_month}.csv'
print(tradefile)
# Read the data from a CSV file or create a DataFrame directly if the data is available in a different format
data = pd.read_csv(tradefile)


# Specify the start and end dates for the time period

start_date = "2023-05-01"
end_date = "2023-06-02"

# Initialize variables
total_investment = 0
total_profit_loss = 0

# Filter the data based on the specified time period
filtered_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

print(filtered_data)

# Iterate over the rows in the filtered DataFrame
for _, row in data.iterrows():
    action = row["Action"]
    price = row["Price"]
    shares = row["Shares"]

    if action == "Buy":
        trade_value = price * shares
        taxes = row["Taxes"]
        cost = trade_value + taxes
        total_investment += cost
        total_profit_loss -= cost
    elif action == "Sell":
        trade_value = price * shares
        taxes = row["Taxes"]
        cost = trade_value - taxes
        total_profit_loss += cost

print(total_profit_loss)
# Calculate the percentage
percentage = (total_profit_loss / total_investment) * 100

# Print the total profit or loss and the percentage
print("Total Profit/Loss: ", total_profit_loss)
print("Percentage: ", percentage, "%")


