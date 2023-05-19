import pandas as pd
import time
from datetime import datetime, timedelta
import yfinance as yf
import os
import numpy as np

# Initialize a DataFrame to hold trade information
trades = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Shares'])

# Initialize a variable to hold total profit or loss
total_profit_or_loss = 0

# Set the fixed amount to trade
current_balance = 1000000
balance_filename = 'balance.txt'
desired_number_of_stocks = 10

# Check if the balance file exists
if os.path.exists(balance_filename):
    # Load the balance from the file
    with open(balance_filename, 'r') as file:
        current_balance = float(file.read())

# Set stop-loss and sell target multipliers
stop_loss_multiplier = 0.98
sell_target_multiplier = 1.08

portfolio_filepath = f'Portfolios/portfolio.csv'

# Check if the file exists
if os.path.exists(portfolio_filepath):
    # Load portfolio from the CSV file
    portfolio_df = pd.read_csv(portfolio_filepath, index_col=0)
    # Convert the DataFrame back into a dictionary
    portfolio = portfolio_df.to_dict(orient='index')
else:
    # If the file doesn't exist, initialize an empty portfolio
    portfolio = {}

today = datetime.today().strftime('%Y-%m-%d')
filepath=f'Analysis/result_{today}.csv'
hist_data = pd.read_csv(filepath)

# Calculate the amount to spend on each trade based on current balance and number of stocks to hold
amount_per_trade = 8000 # current_balance / desired_number_of_stocks
Scancounter=1

def scan_and_sleep(isscanning, stocknumber, totalstocks):
    if(isscanning):
        print(f"Scanning stock {stocknumber}/{totalstocks}", end ='\r')
    else:
        print(".....Sleeping........", end='\r')


def computeRSI (data, time_window):
    diff = data.diff(1).dropna()    

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    # down change is equal to negative difference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

totalstocks= len(hist_data)
while True:
    try:
        # Iterate over the data
        for index, row in hist_data.iterrows():

            scan_and_sleep(True,index,totalstocks)

            # Calculate moving averages and RSI
            ticker = yf.Ticker(row['Symbol'])
            data = ticker.history(period="2d",interval="15m")
            data['RSI'] = computeRSI(data['Close'], 14)
            data['EMA_short'] = data['Close'].ewm(span=6, adjust=False).mean()  # 9-day EMA
            data['EMA_long'] = data['Close'].ewm(span=12, adjust=False).mean()  # 21-day EMA

            latestprice = data["Close"].iloc[-1]
            latest_ema_short = data['EMA_short'].iloc[-1]
            latest_ema_long = data['EMA_long'].iloc[-1]
            latest_rsi = data['RSI'].iloc[-1]

        
            # Check if we own this stock
            if row['Symbol'] in portfolio:

                # Define sell conditions
                is_rsi_overbought = latest_rsi > 70
                is_price_above_sell_target = latestprice >= portfolio[row['Symbol']]['sell_target']
                is_price_below_stop_loss = latestprice <= portfolio[row['Symbol']]['stop_loss']
                is_ema_crossdown = latest_ema_short < latest_ema_long
                
                sell_condition = is_rsi_overbought or is_price_above_sell_target or is_price_below_stop_loss or is_ema_crossdown

                # Update the trailing stop loss
                portfolio[row['Symbol']]['stop_loss'] = max(portfolio[row['Symbol']]['stop_loss'], latestprice * stop_loss_multiplier)

                # Check if the current price is below our stop-loss price or above our sell target price
                if sell_condition:
                    # Sell the stock
                    trades = pd.concat([trades, pd.DataFrame({'Symbol': [row['Symbol']], 'Date': [datetime.now()], 'Action': ['Sell'], 'Price': [latestprice], 'Shares': [desired_number_of_stocks]})], ignore_index=True)
                    # Calculate the profit or loss
                    profit_or_loss = (latestprice - portfolio[row['Symbol']]['buy_price']) * desired_number_of_stocks
                    current_balance += profit_or_loss
                    print(f"sell Trade of {row['Symbol']} price {latestprice} shares {desired_number_of_stocks} total: {profit_or_loss} current balance: {current_balance}")
                    # Remove the stock from our portfolio
                    del portfolio[row['Symbol']]
            else:
                    
                    # Define buy conditions
                    is_rsi_in_range = 30 < latest_rsi < 70
                    is_ema_cross = latest_ema_short > latest_ema_long and row['Close'] < latestprice
                    is_rsi_less_than_30 = latest_rsi < 30 and row['Close'] < latestprice

                    buy_condition1 = is_rsi_in_range and is_ema_cross
                    buy_condition2 = is_rsi_less_than_30
                    buy_condition = buy_condition1 or buy_condition2


                    # Buy condition: if the close price is greater than stop loss by 10% and less than sell target
                    if buy_condition:
                        #Check if current balance value is higher then required Amount
                        if(current_balance > (latestprice * desired_number_of_stocks)):
                            # Check if we have enough balance before buying
                            trades = pd.concat([trades, pd.DataFrame({'Symbol': [row['Symbol']], 'Date': [datetime.now()], 'Action': ['Buy'], 'Price': [latestprice], 'Shares': [desired_number_of_stocks]})], ignore_index=True)
                            # Add the stock to our portfolio and set the stop-loss and sell target prices
                            portfolio[row['Symbol']] = {'buy_price': latestprice, 'stop_loss': latestprice * stop_loss_multiplier, 'sell_target': latestprice * sell_target_multiplier}
                            # Subtract the cost of the purchase from our balance
                            current_balance -= (latestprice * desired_number_of_stocks)
                            print(f"Buy Trade of {row['Symbol']} price {latestprice} shares {desired_number_of_stocks} total: {(latestprice * desired_number_of_stocks)} current balance: {current_balance}")


        # Save trade information to CSV with today's date appended to the filename
        today = datetime.today().strftime('%Y-%m-%d')
        tradefile=f'Trades/trades_{today}.csv'
        if os.path.exists(tradefile):
            existing_trades = pd.read_csv(tradefile)
            all_trades = pd.concat([existing_trades, trades], ignore_index=True)
            all_trades.to_csv(tradefile, index=False)
        else:
            trades.to_csv(tradefile,index=False)

        # Save the updated balance to the file
        with open(balance_filename, 'w') as file:
            file.write(str(current_balance))

        # Clear the trades DataFrame and total_profit_or_loss for the next day
        trades = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Shares'])
        total_profit_or_loss = 0

        # Save portfolio details into a CSV file with today's date appended to the filename
        portfolio_df = pd.DataFrame(portfolio).T  # Transpose the portfolio dictionary to make it suitable for a DataFrame
        portfolio_df.to_csv(f'Portfolios/portfolio.csv', index=True)
        
        scan_and_sleep(False,0,0)

        # Wait for 1 minutes (60)
        time.sleep(60)
    except Exception as e:
        print(f"An error occurred: {e}")


