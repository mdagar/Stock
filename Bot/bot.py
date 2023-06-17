import pandas as pd
from datetime import datetime
import yfinance as yf
import talib as ta
from utility import *
import multiprocessing
import sys

# Initialize a DataFrame to hold trade information
trades = pd.DataFrame(columns=['Symbol', 'Date', 'Action', 'Price', 'Shares','TradeValue','Taxes'])

investpercentage= 0.05
portfolio = {}

if(len(sys.argv)>1):
    ChecktotalBalance = eval(sys.argv[1].capitalize())    
else:
    ChecktotalBalance= get_variable("CheckBalance")

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
    data['EMA_short'] = data['Close'].ewm(span=6, adjust=False).mean()  # 9-day EMA
    data['EMA_long'] = data['Close'].ewm(span=9, adjust=False).mean()  # 21-day EMA
    data['DIp'] = ta.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['DIn'] = ta.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['ADXR'] = ta.ADXR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return data

def get_buy_sell_signals(data, row, transaction_type,portfolio):

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
        sell_condition = (sell_condition1 and is_dmi_bearish and is_adx_strong) or is_price_below_stop_loss

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
        buy_condition = buy_condition1 and is_dmi_bullish and is_adx_strong  and  row['Close'] < latestprice and is_slope_up
        result = buy_condition
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

def trade(bucket,current_balance,investpercentage,ClosePortfolio,tickerlist):
    
    portfolio_filepath = f'Portfolios/portfolio_{bucket}.csv'
    balance_filename = f"Balance/balance_{bucket}.txt"
    tradefile=f'Trades/trades{bucket}_{year_month}.csv'

    amount_per_trade = current_balance*investpercentage//1
    tickerlist,portfolio,current_balance =load_intials(bucket,current_balance,tickerlist,portfolio_filepath,balance_filename)
    
    totalstocks= len(tickerlist)

    while True:
        try:     
            # Iterate over the data
            for index, row in enumerate(tickerlist):
            
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

                    if buy_condition:
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
           
            scan_and_sleep(False,0,0,sleeptime)
            
        except Exception as e:
             print(f"An error occurred: {repr(e)}")

def main():

        enabled_scan_types = get_enabled_scan_types()
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
            #print(bucket)
            if ChecktotalBalance:
                process = multiprocessing.Process(target=calculate_balance, args=(0,bucket))
            else:
                # Create a new process for each bucket
                process = multiprocessing.Process(target=trade, args=(bucket, Intial_Balance,investpercentage,ClosePortfolio,filtered_stocks))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

if __name__ == '__main__':
    main()
    