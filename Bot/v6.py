import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime, timedelta
from sentiment import GetSentimentScore,ExtractSentiment
from sklearn.linear_model import LinearRegression
import json
import os
import time


#Technical Indicators
def add_technical_indicators(data):
   data['SMA_20'] = ta.SMA(data['Close'], timeperiod=20)
   data['SMA_50'] = ta.SMA(data['Close'], timeperiod=50)
   data['EMA_200'] = ta.EMA(data['Close'], timeperiod=200)
   data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
   data['MACD'], data['MACD_signal'], data['MACD_hist'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
   data['BB_upper'], data['BB_middle'], data['BB_lower'] = ta.BBANDS(data['Close'], timeperiod=20)
   data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
   data['CCI'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
   data['ROC'] = ta.ROC(data['Close'], timeperiod=10)
   data['OBV'] = ta.OBV(data['Close'], data['Volume'])
   data['STOCH_K'], data['STOCH_D'] = ta.STOCH(data['High'], data['Low'], data['Close'])
   data['WilliamsR'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
   data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
   data['CMF'] = ta.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10)
   data['PSAR'] = ta.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
   data['Ichimoku_SpanA'], data['Ichimoku_SpanB'] = ta.MINMAX(data['Close'], timeperiod=52)


def get_candlestickpatterns(data):
    candlestick_patterns = {
        "CDLHAMMER": ta.CDLHAMMER(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLHANGINGMAN": ta.CDLHANGINGMAN(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLENGULFING": ta.CDLENGULFING(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLMORNINGSTAR": ta.CDLMORNINGSTAR(data["Open"], data["High"], data["Low"], data["Close"], penetration=0),
        "CDLEVENINGSTAR": ta.CDLEVENINGSTAR(data["Open"], data["High"], data["Low"], data["Close"], penetration=0),
        "CDLDOJI": ta.CDLDOJI(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLHARAMI": ta.CDLHARAMI(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLSHOOTINGSTAR": ta.CDLSHOOTINGSTAR(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLINVERTEDHAMMER": ta.CDLINVERTEDHAMMER(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLMARUBOZU": ta.CDLMARUBOZU(data["Open"], data["High"], data["Low"], data["Close"]),
        "CDLPIERCING": ta.CDLPIERCING(data["Open"], data["High"], data["Low"], data["Close"]),
    }
    return candlestick_patterns

# Calculate the Buy and Sell signals
def calcuate_signal(data):

    candlestick_patterns =get_candlestickpatterns(data)

    # Add Buy and Sell signals for each pattern
    for pattern, signal in candlestick_patterns.items():
        data[pattern + "_Buy"] = np.where(signal > 0, 1, 0)
        data[pattern + "_Sell"] = np.where(signal < 0, -1, 0)

    data['SMA_Buy'] = np.where(data['SMA_20'] > data['SMA_50'], 1, 0)
    data['SMA_Sell'] = np.where(data['SMA_20'] < data['SMA_50'], -1, 0)

    data['RSI_Buy'] = np.where(data['RSI'] < 30, 1, 0)
    data['RSI_Sell'] = np.where(data['RSI'] > 70, -1, 0)

    data['MACD_Buy'] = np.where(data['MACD'] > data['MACD_signal'], 1, 0)
    data['MACD_Sell'] = np.where(data['MACD'] < data['MACD_signal'], -1, 0)

    data['BB_Buy'] = np.where(data['Close'] < data['BB_lower'], 1, 0)
    data['BB_Sell'] = np.where(data['Close'] > data['BB_upper'], -1, 0)

    data['OBV_Buy'] = np.where(data['OBV'].pct_change() > 0, 1, 0)
    data['OBV_Sell'] = np.where(data['OBV'].pct_change() < 0, -1, 0)

    data['STOCH_Buy'] = np.where(data['STOCH_K'] < 20, 1, 0)
    data['STOCH_Sell'] = np.where(data['STOCH_K'] > 80, -1, 0)

    data['WilliamsR_Buy'] = np.where(data['WilliamsR'] < -80, 1, 0)
    data['WilliamsR_Sell'] = np.where(data['WilliamsR'] > -20, -1, 0)

    data['ROC_Buy'] = np.where(data['ROC'] > 0, 1, 0)
    data['ROC_Sell'] = np.where(data['ROC'] < 0, -1, 0)

    data['CCI_Buy'] = np.where(data['CCI'] < -100, 1, 0)
    data['CCI_Sell'] = np.where(data['CCI'] > 100, -1, 0)

    # Calculate total Buy and Sell signals for all patterns
    data["CDL_Buy"] = sum(data[pattern + "_Buy"] for pattern in candlestick_patterns.keys())
    data["CDL_Sell"] = sum(data[pattern + "_Sell"] for pattern in candlestick_patterns.keys())

    # Calculate buy and sell signals for ADX
    data['ADX_Buy'] = np.where(data['ADX'] > 25, 1, 0)
    data['ADX_Sell'] = np.where(data['ADX'] < 20, -1, 0)

    data['CMF_Buy'] = np.where(data['CMF'] > 0, 1, 0)
    data['CMF_Sell'] = np.where(data['CMF'] < 0, -1, 0)

    data['PSAR_Buy'] = np.where(data['PSAR'] < data['Close'], 1, 0)
    data['PSAR_Sell'] = np.where(data['PSAR'] > data['Close'], -1, 0)

    data['Ichimoku_Buy'] = np.where(data['Close'] > ((data['Ichimoku_SpanA'] + data['Ichimoku_SpanB']) / 2), 1, 0)
    data['Ichimoku_Sell'] = np.where(data['Close'] < ((data['Ichimoku_SpanA'] + data['Ichimoku_SpanB']) / 2), -1, 0)

    #data.to_csv('data.csv', index=False)  # Saves the DataFrame as a CSV file without including the index


def getnewssentiments(data,ticker):

    sentimentdata= GetSentimentScore(ticker)
    sentimentdata['sentiment'] = pd.to_numeric(sentimentdata['sentiment'], errors='coerce')
    sentimentdata = sentimentdata.groupby('date')['sentiment'].sum().reset_index()

    # Convert the Series to a DataFrame
    sentimentdata = pd.DataFrame(sentimentdata)
    sentimentdata.columns = ['date', 'NewsSentiment']

    data['date'] = pd.to_datetime(data['Date'])
    sentimentdata['date'] = pd.to_datetime(sentimentdata['date'])

    data = pd.merge(data, sentimentdata, on='date', how='left')
    
    #data = data.drop('date', axis=1)
    # Convert the 'sentiment' column to numeric type
    data['NewsSentiment'] = pd.to_numeric(data['NewsSentiment'], errors='coerce')
    #data['NewsSentiment'].fillna(method='ffill', inplace=True)
    data['NewsSentiment'].fillna(0)

    #print(data['NewsSentiment'])
    data['NewsSentiment'].fillna(0, inplace=True)
    data['NewsSentiment'] = data['NewsSentiment'].round(2)
    return data


def calculate_signalstrength(data):
    
    indicator_weights = {
        'SMA': None,'RSI': None,
        'MACD': None,'BB': None, 'OBV': None,'STOCH': None, 'WilliamsR': None, 'CCI': None,
        'ROC': None, 'Candlestick': None,'ADX': None, 'CMF': None,'Ichimoku': None, 'PSAR': None,
        'StockSentiment': None
    }
    # Calculate the weight based on the number of indicators
    weight = 1.0 / len(indicator_weights)
    # Update the weights
    for key in indicator_weights:
        indicator_weights[key] = weight

    # Calculate signal strength
    data['Signal_strength'] = (
        indicator_weights['SMA'] * (data['SMA_Buy'] + data['SMA_Sell']) +
        indicator_weights['RSI'] * (data['RSI_Buy'] + data['RSI_Sell']) +
        indicator_weights['MACD'] * (data['MACD_Buy'] + data['MACD_Sell']) +
        indicator_weights['BB'] * (data['BB_Buy'] + data['BB_Sell']) +
        indicator_weights['OBV'] * (data['OBV_Buy'] + data['OBV_Sell']) +
        indicator_weights['STOCH'] * (data['STOCH_Buy'] + data['STOCH_Sell']) +
        indicator_weights['WilliamsR'] * (data['WilliamsR_Buy'] + data['WilliamsR_Sell']) +
        indicator_weights['CCI'] * (data['CCI_Buy'] + data['CCI_Sell']) +
        indicator_weights['ROC'] * (data['ROC_Buy'] + data['ROC_Sell']) +
        indicator_weights['StockSentiment'] * (data['NewsSentiment']) +
        indicator_weights['ADX'] * (data['ADX_Buy'] + data['ADX_Sell'])+
        indicator_weights['CMF'] * (data['CMF_Buy'] + data['CMF_Sell'])+
        indicator_weights['Ichimoku'] * (data['Ichimoku_Buy'] + data['Ichimoku_Sell'])+
        indicator_weights['PSAR'] * (data['PSAR_Buy'] + data['PSAR_Sell'])+
        indicator_weights['Candlestick'] * (data['CDL_Buy'] + data['CDL_Sell'])
    )

def cleardata(data):
    data['Signal_strength'] = data['Signal_strength'].round(2)
    data['Close'] = data['Close'].round(2)
    data['High'] = data['High'].round(2)
    data['Low'] = data['Low'].round(2)
    data.dropna(subset=['High', 'Low', 'Close'], inplace=True)

def Analysis(symbol):

    end = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=300)).strftime('%Y-%m-%d')
    
    #sentimentscore= GetSentimentScore(symbol)
    #start = '2020-03-01'
    #end = '2023-05-12'

    data = yf.download(symbol, progress=False, prepost=False, start=start, end=end)
    
    data = data.reset_index()
    data.rename(columns={'index': 'Date'}, inplace=True)
    add_technical_indicators(data)

    data= getnewssentiments(data,symbol)
    calcuate_signal(data)
    calculate_signalstrength(data)
    cleardata(data)
    return data

def ScanSignals(data,symbol,strenghtfilter):

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')
    # calculate next day close price
    data['NextDayClose'] = data['Close'].shift(-1)
    # calculate the percentage change in price
    data['PercentChange'] = ((data['NextDayClose'] - data['Close']) / data['Close']) * 100
    strong_signals = data[(data['Signal_strength']>0) & (data['PercentChange'] > 0)]

    # Latest Signal on data
    last_row_signal_strength = data['Signal_strength'].iloc[-1]
    #print(strong_signals[['Date','NewsSentiment','Signal_strength','Close','NextDayClose','PercentChange','High','Low']].tail(10))
    min_signal_strength= strong_signals["Signal_strength"].min()
    
    data['MinSignalStrength'] = min_signal_strength
    # Define a rolling window size
    window_size = 25
    # Calculate rolling max and min within the window size to get resistance and support
    data['Resistance'] = data['Close'].rolling(window_size).max()
    data['Support'] = data['Low'].rolling(window_size).min()
    yearhigh= data['Close'].max()
    yearlow= data['Close'].min()
    data['52High']= yearhigh
    data['52Low']= yearlow

    data['Cum_Vol'] = data['Volume'].cumsum()
    data['Cum_Vol_Price'] = (data['Close']*data['Volume']).cumsum()
    data['VWAP'] = data['Cum_Vol_Price'] / data['Cum_Vol']

    # Define the trend
    data['Trend'] = 'Neutral'
    data.loc[(data['SMA_20'] > data['SMA_50']), 'Trend'] = 'Bullish'
    data.loc[(data['SMA_20'] < data['SMA_50']), 'Trend'] = 'Bearish'
    data.loc[:, 'Symbol'] = symbol

    today = datetime.today().strftime('%Y-%m-%d')
    filepath=f'Analysis/result_{today}.csv'
    if os.path.exists(filepath):
        existingdata = pd.read_csv(filepath)

    if(last_row_signal_strength >= min_signal_strength and last_row_signal_strength>=strenghtfilter ):

        # calculate_target_price_technical_analysis(data)
                
        selected_columns = ['Symbol','Date','NewsSentiment','Signal_strength','MinSignalStrength','Close','High','Low','Resistance','Support','52High','52Low','VWAP','Trend']        
        df_selected = data[selected_columns].tail(1).copy()
        print(data[selected_columns].tail(1))
        print("-----------------------------------------------------------------------------------------------------")
        if not os.path.exists(filepath):
            df_selected.to_csv(filepath, index=False)
        else:
            check_columns = ['Symbol','Date']
            row_values = df_selected[check_columns].iloc[0].to_dict()
            row_exists = existingdata[check_columns].eq(row_values).all(axis=1).any()
            if not row_exists:
                df_selected.to_csv(filepath, mode='a',index=False, header=False)
            #check_columns = ['Symbol','Date', 'NewsSentiment', 'Signal_strength']
            #for now check only date and symbol later based on startey we will update the filter 


def calculate_target_price_technical_analysis(data):

    # stock = yf.Ticker(ticker)
    # hist = stock.history(period="1y") # Get historical data for the past year
    hist= data.copy() 
    hist = hist.reset_index()
    hist = hist[['Date', 'Close']]
    hist['Date'] = hist.index
    model = LinearRegression()
    model.fit(hist.Date.values.reshape(-1, 1), hist.Close.values)
    target_price = model.predict(np.array([hist.Date.values[-1] + 1]).reshape(-1, 1))[0]
    data['TargetP']= target_price
    #print(f'Target Price {ticker}: using technical analysis {target_price}')
    

def showothersentiments():
    print(GetSentimentScore("Buiness"))
    print(GetSentimentScore("India"))
    print(GetSentimentScore("World"))

def ExtractOtherSeniments():
    newsextractcount=10
    ExtractSentiment("Buiness","Buiness",newsextractcount)
    ExtractSentiment("India","India",newsextractcount)
    ExtractSentiment("World","World",newsextractcount)


def main():

    start_time = time.time()
    extractnews= True
    runAnalysis= False
    extraxtglobalnews= False
    strenghtfilter=0.2

    scantype="Indianstocks"

    with open('s.json') as file:
      stocklist = json.load(file)

    newsextractcount= 5

    if(extraxtglobalnews):
        ExtractOtherSeniments()

    if(extractnews):
        for stock in stocklist[scantype]:
            print(f"getting news of {stock['name']} .")
            ExtractSentiment(stock["symbol"],stock["name"],newsextractcount)

    if(runAnalysis):
        for stock in stocklist[scantype]:
                symbol = stock["symbol"]
                #print(symbol)
                Query = stock["name"]
                df= Analysis(symbol)   
                ScanSignals(df,symbol,strenghtfilter)
                #calculate_target_price_technical_analysis(symbol)
    
    end_time = time.time()
    # Calculate total time taken in seconds
    total_time = end_time - start_time
    # Convert total time to minutes and seconds
    minutes, seconds = divmod(total_time, 60)
    print(f"The process took {int(minutes)} minute(s) and {int(seconds)} seconds.")

if __name__ == "__main__":
    main()

