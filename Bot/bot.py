import pandas as pd
from datetime import datetime
import time
import yfinance as yf
import talib as ta
from utility import *
import multiprocessing
import sys
import pytz


# Initialize a DataFrame to hold trade information
trades = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Shares','TradeValue','Taxes'])

investpercentage= 0.05
portfolio = {}

if(len(sys.argv)>1):
    RunBot = eval(sys.argv[1].capitalize())    
else:
    RunBot= get_variable("RunBot")

stop_loss_multiplier = get_variable("stop_loss_multiplier")
sell_target_multiplier = get_variable("sell_target_multiplier")
sleeptime= get_variable("sleeptime")

# Add constants for each type of charge
BROKERAGE_RATE = get_variable("BROKERAGE_RATE")  
SEBI_CHARGE_RATE = get_variable("SEBI_CHARGE_RATE")
TRANSACTION_CHARGE_RATE = get_variable("TRANSACTION_CHARGE_RATE")
GST_RATE = get_variable("GST_RATE")
STAMP_DUTY_RATE = get_variable("STAMP_DUTY_RATE")
#balance_filename = f"Balance/balance.txt"
today = datetime.today().strftime('%Y-%m-%d')
year_month = datetime.today().strftime('%Y')
#tradefile=f'Trades/trades{}_{year_month}.csv'
prev_ema = {}
time_period= 15

# file paths
filepath=f'Analysis/result_{today}.csv'

def calculate_technical_indicators(data):
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    data['EMA_short'] = data['Close'].ewm(span=9, adjust=False).mean()  # 9-day EMA
    data['EMA_long'] = data['Close'].ewm(span=26, adjust=False).mean()  # 21-day EMA
    data['DIp'] = ta.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['DIn'] = ta.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['ADXR'] = ta.ADXR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_Upper'], data['BB_middle'], data['BB_Lower'] = ta.BBANDS(data['Close'], timeperiod=26)
    data['WilliamR'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return data

def get_buy_sell_signals(data, row, transaction_type,portfolio):
    result = True
    latest_dip = data['DIp'].iloc[-1]
    latest_din = data['DIn'].iloc[-1]
    latest_adx = data['ADX'].iloc[-1]
    WilliamR = data['WilliamR'].iloc[-1]
    is_WilliamR_signal_in_range= False

    if WilliamR > -30 or WilliamR < -75:
        is_WilliamR_signal_in_range= True
  
    latest_macd = data['MACD'].iloc[-1]
    latest_signal = data['MACD_Signal'].iloc[-1]
    upper_band = data['BB_Upper'].iloc[-1]
    lower_band = data['BB_Lower'].iloc[-1]
    latest_close = data['Close'].iloc[-1]

    is_dmi_bullish = latest_dip > latest_din
    is_dmi_bearish = latest_dip < latest_din
    is_adx_strong = latest_adx > 25
    latestprice = data["Close"].iloc[-1]
    latest_ema_short = data['EMA_short'].iloc[-1]
    latest_ema_long = data['EMA_long'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    middle_band = data['BB_middle'].iloc[-1]
    is_macd_bearish = latest_macd < latest_signal
    is_touching_upper_band = latest_close >= upper_band
 
    if row['Symbol'] in prev_ema:
        slope_ema_short = (latest_ema_short - prev_ema[row['Symbol']]) / time_period
    else:
        slope_ema_short = 0  # or some other default value
    prev_ema[row['Symbol']] = latest_ema_short

    is_buy_conditon= False
    is_sell_conditon= False

    if(transaction_type=="Sell"):
        new_stop_loss = stop_loss_multiplier * latestprice  # proposed new stop loss
        if new_stop_loss > portfolio[row['Symbol']]['stop_loss']:
            portfolio[row['Symbol']]['stop_loss'] = new_stop_loss 
        if latestprice > portfolio[row['Symbol']]['sell_target']:
            portfolio[row['Symbol']]['sell_target'] = 1.005 * latestprice  # update target to 0.5% of latest price

        is_price_below_stop_loss = latestprice <= portfolio[row['Symbol']]['stop_loss']
        is_sell_conditon=is_price_below_stop_loss
    
    # Check buy conditions
    if(latestprice < row["Close"]):
        is_buy_conditon = False
    else:
        is_ema_cross = latest_ema_short > latest_ema_long
        is_slope_up = slope_ema_short > 0
        is_macd_cross = latest_macd > latest_signal
        is_above_bollinger = middle_band <= latest_close
        buy_condition1 = (is_ema_cross) and is_slope_up
        buy_condition = buy_condition1 and is_dmi_bullish and is_adx_strong and is_macd_cross and is_above_bollinger and is_WilliamR_signal_in_range
        is_buy_conditon = buy_condition and (is_touching_upper_band==False)
    if(transaction_type=="Sell"):
        if(is_sell_conditon == True and is_buy_conditon == False):
            result= True
        else:
            result= False
    else :
        result= is_buy_conditon
    if(row['Symbol'] == "SYRMA.ns"):
        print(result)
    return result
        

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

def add_portfolio_symbols(portfolio, scaned_symbols):
    # If portfolio is empty, return the original scaned_symbols list
    if not portfolio:
        return scaned_symbols

    # If scaned_symbols is empty, initialize it with the symbols from the portfolio
    if not scaned_symbols:
        scaned_symbols = [{'Symbol': symbol, 'Close': 0, 'Bucket': portfolio[symbol]['Bucket']} for symbol in portfolio.keys()]
        return scaned_symbols

    for symbol in portfolio.keys():
        if symbol not in [item['Symbol'] for item in scaned_symbols]:
            scaned_symbols.append({'Symbol': symbol, 'Close': 0, 'Bucket': portfolio[symbol]['Bucket']})
    return scaned_symbols

def get_mkt_direction(indexsymbol='^NSEI'):
    mkt_direction= True
    try:
        data = yf.download(indexsymbol, period='1d', progress=False)  
        opening_price = data['Open'].iloc[0]
        current_price = data['Close'].iloc[-1]
        change = current_price - opening_price
        percentage_change = (change / opening_price) * 100
        if percentage_change <= -0.10:
            mkt_direction = False
    except Exception as e:
        print(f"An error occurred: {repr(e)}")
    return mkt_direction

def saveportfolio(portfolio,portfolio_filepath):
        portfolio_df = pd.DataFrame(portfolio).T
        portfolio_df.to_csv(portfolio_filepath, index=True)

def load_intials(bucket,current_balance,tickerlist,portfolio_filepath,balance_filename):
    #Validate portfolio
    current_balance=load_balance(balance_filename,current_balance)
    portfolio=load_portfolio(portfolio_filepath)
    final_symbols=add_portfolio_symbols(portfolio,tickerlist)
    return final_symbols,portfolio,current_balance

analysed_symbols = pd.read_csv(filepath)
dict_selected = analysed_symbols[['Symbol', 'Close','Bucket']].to_dict('records')


def calculate_balance(c,bucket):
    
    portfolio_filepath = f'Portfolios/portfolio_{bucket}.csv'
    balance_filename = f"Balance/balance_{bucket}.txt"
    tradefile=f'Trades/trades{bucket}_{year_month}.csv'

    current_balance=0
    tickerlist,portfolio,current_balance =load_intials(bucket,current_balance,{},portfolio_filepath,balance_filename)
    
    total_value = 0
    for symbol in portfolio.keys():
        #print(symbol)
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
        total_value += portfolio[symbol]['Quantity'] * current_price
    print(f" Bucket {bucket} Cash: {current_balance} Stock {total_value} Total Balance is: {total_value+current_balance}")

def trade(bucketindex,bucket,current_balance,investpercentage,ClosePortfolio,tickerlist):
    
    portfolio_filepath = f'Portfolios/portfolio_{bucket}.csv'
    balance_filename = f"Balance/balance_{bucket}.txt"
    tradefile=f'Trades/trades{bucket}_{year_month}.csv'

    amount_per_trade = current_balance*investpercentage//1
    tickerlist,portfolio,current_balance =load_intials(bucket,current_balance,tickerlist,portfolio_filepath,balance_filename)
    
    totalstocks= len(tickerlist)
    ist = pytz.timezone('Asia/Kolkata')

    while True:
        try:     
            current_time = datetime.now(ist)
            # If it's before 9:15 AM, print market not started
            if (current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 15)):
                market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
                time_to_wait = (market_open_time - current_time).total_seconds()
                print("Market not started yet")
                scan_and_sleep(False,0,totalstocks,int(time_to_wait))
                continue
            if (current_time.hour > 15 or (current_time.hour == 15 and current_time.minute >= 29)):
                print("\n Market closed.......")
                break

            mkt_direction= get_mkt_direction(bucketindex)

            # Iterate over the data
            for index, row in enumerate(tickerlist):
                try:
                    scan_and_sleep(True,index,totalstocks,sleeptime)
                    ticker = yf.Ticker(row['Symbol'])
                    data = ticker.history(period="2d",interval="15m")
                    #print(data.tail())
                    data =calculate_technical_indicators(data)

                    latestprice = data["Close"].iloc[-1]
                    levels = get_fibonacci_levels(data)

                    if row['Symbol'] in portfolio:

                        if(ClosePortfolio== True):
                            sell_condition = True
                        else:
                            sell_condition = get_buy_sell_signals(data,row,"Sell",portfolio)

                        portfolio[row['Symbol']]['stop_loss'] = max(portfolio[row['Symbol']]['stop_loss'], (latestprice * stop_loss_multiplier),levels["support"])
                        
                        # Check if the current price is below our stop-loss price or above our sell target price
                        if sell_condition:

                            Amount=(latestprice*portfolio[row['Symbol']]['Quantity'])
                            Additional_charges=calculate_charges(Amount,"sell")

                            profit_or_loss = ((latestprice - portfolio[row['Symbol']]['buy_price']) * portfolio[row['Symbol']]['Quantity'])- Additional_charges
                            _trades = pd.DataFrame({'Bucket':bucket,'Symbol': [row['Symbol']], 'Date': [datetime.now()], 'Action': ['Sell'], 'Price': [latestprice], 'Shares': [portfolio[row['Symbol']]['Quantity']],'TradeValue':[Amount],'Taxes':[Additional_charges]}) 
                            
                            current_balance += (Amount-Additional_charges)
                            save_trade(_trades,tradefile)

                            print(f"sell {bucket}  {row['Symbol']} price {latestprice} shares {portfolio[row['Symbol']]['Quantity']} total: {profit_or_loss} current balance: {current_balance}")
                            del portfolio[row['Symbol']]
                            del prev_ema[row['Symbol']]
                    else:
                            
                        if(ClosePortfolio== True):
                            buy_condition = False 
                        else:
                            buy_condition = get_buy_sell_signals(data,row,"Buy",portfolio)

                        if buy_condition and mkt_direction:
                            desired_number_of_stocks = (amount_per_trade/latestprice)//1
                            stock_amount= (latestprice * desired_number_of_stocks)
                            if(current_balance > stock_amount and desired_number_of_stocks>0):
                                
                                Additional_charges=calculate_charges(stock_amount,"buy")
                                _trades = pd.DataFrame({'Bucket':bucket,'Symbol': [row['Symbol']], 'Date': [datetime.now()], 'Action': ['Buy'], 'Price': [latestprice], 'Shares': [desired_number_of_stocks],'TradeValue':[stock_amount],'Taxes':[Additional_charges]})
                                save_trade(_trades,tradefile)
        
                                portfolio[row['Symbol']] = {'buy_price': latestprice, 'stop_loss': latestprice * stop_loss_multiplier, 'sell_target': latestprice * sell_target_multiplier,'Quantity':desired_number_of_stocks,'Bucket':row['Bucket']}
                                #print(portfolio)
                                current_balance -= stock_amount+Additional_charges
                                print(f"Buy {bucket} {row['Symbol']} price {latestprice} shares {desired_number_of_stocks} total: {(latestprice * desired_number_of_stocks)} current balance: {current_balance}")

                    saveportfolio(portfolio,portfolio_filepath)
                    save_balance(current_balance,balance_filename)
                except Exception as e:
                     print(f"An error occurred: {repr(e)}")

            scan_and_sleep(False,0,0,sleeptime)
            
        except Exception as e:
             print(f"An error occurred: {repr(e)}")

def main():

        enabled_scan_types = get_enabled_scan_types()
        # print(enabled_scan_types)
        buckets = []


        if enabled_scan_types:
            for scan_type in enabled_scan_types:
                buckets.append(scan_type["bucket"])

        processes = []
        for bucket in buckets:
            
            bucket_obj = [obj for obj in enabled_scan_types if obj['bucket'] == bucket]
            
            # Filter symbols based on the bucket
            filtered_stocks = [item for item in dict_selected if item['Bucket'] == bucket]


            ClosePortfolio = bucket_obj[0]['ClosePortfolio']
            Intial_Balance = bucket_obj[0]['Intial_Balance']
            investpercentage = bucket_obj[0]['investpercentage']
            bucketindex = bucket_obj[0]['Index']

            #print(bucket)
            if not RunBot:
                process = multiprocessing.Process(target=calculate_balance, args=(0,bucket))
            else:
                # Create a new process for each bucket
                process = multiprocessing.Process(target=trade, args=(bucketindex,bucket, Intial_Balance,investpercentage,ClosePortfolio,filtered_stocks))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

if __name__ == '__main__':
    main()
    