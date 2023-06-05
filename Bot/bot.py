import pandas as pd
import time
from datetime import datetime, timedelta
import yfinance as yf
import os
import numpy as np
from pytz import timezone
import talib as ta
from utility import *

# Initialize a DataFrame to hold trade information
trades = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Shares','TradeValue','Taxes'])
# Set the fixed amount to trade
current_balance = 30000000
investpercentage= 0.05
amount_per_trade= current_balance*investpercentage
portfolio = {}
ChecktotalBalance= get_variable("CheckBalance")

index, bucket = get_scan_type()
stop_loss_multiplier = 0.98
sell_target_multiplier = 1.01
today = datetime.today().strftime('%Y-%m-%d')
year_month = datetime.today().strftime('%Y')
prev_ema = {}
time_period= 15
sleeptime= get_variable("sleeptime")

# file paths
filepath=f'Analysis/{bucket}_result_{today}.csv'
portfolio_filepath = f'Portfolios/{bucket}_portfolio.csv'
balance_filename = f"Balance/balance.txt"
tradefile=f'Trades/trades_{year_month}.csv'

def save_trade(trades):
    # Save trade information to CSV with today's date appended to the filename
    if os.path.exists(tradefile):
        existing_trades = pd.read_csv(tradefile)
        all_trades = pd.concat([existing_trades, trades], ignore_index=True)
        all_trades.to_csv(tradefile, index=False)
    else:
        trades.to_csv(tradefile,index=False)     

def save_balance(current_balance):
    # Save the updated balance to the file
    with open(balance_filename, 'w') as file:
       file.write(str(current_balance))

def calculate_technical_indicators(data):
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    data['EMA_short'] = data['Close'].ewm(span=6, adjust=False).mean()  # 9-day EMA
    data['EMA_long'] = data['Close'].ewm(span=9, adjust=False).mean()  # 21-day EMA
    data['DIp'] = ta.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['DIn'] = ta.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['ADXR'] = ta.ADXR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return data

def get_buy_sell_signals(data, row, transaction_type):

    result = True
    latest_dip = data['DIp'].iloc[-1]
    latest_din = data['DIn'].iloc[-1]
    latest_adx = data['ADX'].iloc[-1]

    is_dmi_bullish = latest_dip > latest_din
    is_dmi_bearish = latest_dip < latest_din
    is_adx_strong = latest_adx > 25
    latestprice = data["Close"].iloc[-1]
    latest_ema_short = data['EMA_short'].iloc[-1]
    latest_ema_long = data['EMA_long'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]

    #print(f"Stock {row['Symbol']}, DIP: {latest_dip} DIN: {latest_din} ADX: {latest_adx}")

    if row['Symbol'] in prev_ema:
        slope_ema_short = (latest_ema_short - prev_ema[row['Symbol']]) / time_period
    else:
        slope_ema_short = 0  # or some other default value
        prev_ema[row['Symbol']] = latest_ema_short

    if(transaction_type=="Sell"):

        is_rsi_overbought = latest_rsi > 70
        is_price_above_sell_target = latestprice >= portfolio[row['Symbol']]['sell_target']
        is_price_below_stop_loss = latestprice <= portfolio[row['Symbol']]['stop_loss']
        is_ema_crossdown = latest_ema_short < latest_ema_long   
        is_slope_down = slope_ema_short < 0           
        sell_condition1 = (is_rsi_overbought or is_price_above_sell_target or is_price_below_stop_loss or is_ema_crossdown) and is_slope_down
        sell_condition = (sell_condition1 and is_dmi_bearish and is_adx_strong)

        # print(f"RSI over: {is_rsi_overbought} , Price above Tgt: {is_price_above_sell_target} below stoploss: {is_price_below_stop_loss} ")
        # print(f"Ema down {is_ema_crossdown} ")
        # print(f"----- {sell_condition}")

        result= sell_condition
    else:
        # Define buy conditions
        is_rsi_in_range = 30 < latest_rsi < 70
        is_ema_cross = latest_ema_short > latest_ema_long
        is_rsi_less_than_30 = latest_rsi < 30 
        is_slope_up = slope_ema_short > 0
        buy_condition1 = (is_rsi_in_range and is_ema_cross) or is_rsi_less_than_30
        buy_condition = buy_condition1 and is_dmi_bullish and is_adx_strong  and  row['Close'] < latestprice # and is_slope_up
        result = buy_condition
    return result

# Add constants for each type of charge
BROKERAGE_RATE = 0.0003 
SEBI_CHARGE_RATE = 0.00000001 # 0.000001% 
TRANSACTION_CHARGE_RATE = 0.0000375
GST_RATE = 0.18  # 18%
STAMP_DUTY_RATE = 0.00015 

def calculate_charges(amount, transaction_type):
    brokerage = amount * BROKERAGE_RATE
    sebi_charge = amount * SEBI_CHARGE_RATE
    transaction_charge = amount * TRANSACTION_CHARGE_RATE
    # GST is applied to the sum of brokerage, sebi_charge and transaction_charge
    gst = (brokerage + sebi_charge + transaction_charge) * GST_RATE
    stamp_duty = amount * STAMP_DUTY_RATE if transaction_type == 'buy' else 0
    total_charges = brokerage + sebi_charge + gst + transaction_charge + stamp_duty
    return total_charges

def get_fibonacci_levels(df: pd.DataFrame) -> dict:

    # Get the maximum and minimum price
    max_price = df['High'].max()
    min_price = df['Low'].min()

    # Fibonacci Levels considering original trend as upward move
    diff = max_price - min_price
    levels = {
        "resistance": max_price,
        "Level 23.6%": max_price - 0.236 * diff,
        "Level 38.2%": max_price - 0.382 * diff,
        "Level 61.8%": max_price - 0.618 * diff,
        "support": min_price
    }
    return levels

def add_portfolio_symbols(scaned_symbols):
    for symbol in portfolio.keys():
        if symbol not in [item['Symbol'] for item in scaned_symbols]:
            scaned_symbols.append({'Symbol': symbol, 'Close': 0})
    return scaned_symbols

#Validate portfolio
current_balance=load_balance(balance_filename,current_balance)
portfolio=load_portfolio(portfolio_filepath)
analysed_symbols = pd.read_csv(filepath)

dict_selected = analysed_symbols[['Symbol', 'Close']].to_dict('records')
final_symbols=add_portfolio_symbols(dict_selected)
totalstocks= len(final_symbols)

while True:
    try:     

        if(ChecktotalBalance): 
            calculate_balance(portfolio,current_balance)
            break
        
        # Iterate over the data
        for index, row in enumerate(final_symbols):
        
            scan_and_sleep(True,index,totalstocks,sleeptime)

            ticker = yf.Ticker(row['Symbol'])
            data = ticker.history(period="2d",interval="15m")
            #print(data.tail())
            data =calculate_technical_indicators(data)

            latestprice = data["Close"].iloc[-1]
            levels = get_fibonacci_levels(data)

            if row['Symbol'] in portfolio:

                sell_condition = get_buy_sell_signals(data,row,"Sell")                
                portfolio[row['Symbol']]['stop_loss'] = max(portfolio[row['Symbol']]['stop_loss'], (latestprice * stop_loss_multiplier),levels["support"])
                
                # Check if the current price is below our stop-loss price or above our sell target price
                if sell_condition:

                    Amount=(latestprice*portfolio[row['Symbol']]['Quantity'])
                    Additional_charges=calculate_charges(Amount,"sell")

                    profit_or_loss = ((latestprice - portfolio[row['Symbol']]['buy_price']) * portfolio[row['Symbol']]['Quantity'])- Additional_charges

                    trades = pd.concat([trades, pd.DataFrame({'Symbol': [row['Symbol']], 'Date': [datetime.now()], 'Action': ['Sell'], 'Price': [latestprice], 'Shares': [portfolio[row['Symbol']]['Quantity']],'TradeValue':[Amount],'Taxes':[Additional_charges]})], ignore_index=True)
                    current_balance += (Amount-Additional_charges)

                    print(f"sell Trade of {row['Symbol']} price {latestprice} shares {portfolio[row['Symbol']]['Quantity']} total: {profit_or_loss} current balance: {current_balance}")
                    del portfolio[row['Symbol']]
                    del prev_ema[row['Symbol']]
            else:
                buy_condition = get_buy_sell_signals(data,row,"Buy")      
                if buy_condition:
                    desired_number_of_stocks =amount_per_trade/latestprice
                    stock_amount= (latestprice * desired_number_of_stocks)
                    if(current_balance > stock_amount):
                        
                        Additional_charges=calculate_charges(stock_amount,"buy")

                        trades = pd.concat([trades, pd.DataFrame({'Symbol': [row['Symbol']], 'Date': [datetime.now()], 'Action': ['Buy'], 'Price': [latestprice], 'Shares': [desired_number_of_stocks],'TradeValue':[stock_amount],'Taxes':[Additional_charges]})], ignore_index=True)
                        portfolio[row['Symbol']] = {'buy_price': latestprice, 'stop_loss': latestprice * stop_loss_multiplier, 'sell_target': latestprice * sell_target_multiplier,'Quantity':desired_number_of_stocks}
                        
                        current_balance -= stock_amount+Additional_charges

                        print(f"Buy Trade of {row['Symbol']} price {latestprice} shares {desired_number_of_stocks} total: {(latestprice * desired_number_of_stocks)} current balance: {current_balance}")

     
        save_trade(trades)
        save_balance(current_balance)

        # Reset trade object
        trades = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Shares','TradeValue','Taxes'])

        portfolio_df = pd.DataFrame(portfolio).T
        portfolio_df.to_csv(portfolio_filepath, index=True)
        
        scan_and_sleep(False,0,0,sleeptime)
        
    except Exception as e:
        print(f"An error occurred: {e}")
