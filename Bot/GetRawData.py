import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta



# API key for Alpha Vantage
API_KEY = "FDMYC9R1L3EA00N0"



def GetStockList():
    
    return {"MARINE.NS",
            "FCSSOFT.NS",
            "SBC.NS",
            "URJA.NS",
            "WELSPUNIND.NS",
            "RBLBANK.NS",
            "HFCL.NS",
            "CESC.NS",
            "PPLPHARMA.NS","YESBANK.NS","AMARAJABAT.NS"
            }

    # print(symbols)

def yfinance(symbol):

    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=80)).strftime('%Y-%m-%d')
    
    tickerData = yf.Ticker(symbol)
    df = tickerData.history(period='1d', start=start_date, end=end_date)
    #df = tickerData.history(interval='15m', period='1d')
    df= df.iloc[:,0:5]
    df.columns = df.columns.str.lower()
    return df


def GetMonthly(symbol,interval='5min'):

    # Specify the function for the weekly data request
    function = 'TIME_SERIES_MONTHLY'
    #interval='5min'
    # Fetch historical stock price data from Alpha Vantage for the symbol
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={API_KEY}"

    response = requests.get(url)
    # Convert data to pandas DataFrame
    key ='Monthly Time Series'
    data = response.json()[key]
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df;

def GetIntraday(symbol,interval='5min'):

    # Specify the function for the weekly data request
    function = 'TIME_SERIES_INTRADAY'
    #interval='5min'
    # Fetch historical stock price data from Alpha Vantage for the symbol
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&apikey={API_KEY}"

    response = requests.get(url)
    # Convert data to pandas DataFrame
    key ='Time Series ('+interval +')'
    data = response.json()[key]
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df;

def GetTimeSeriesWeekly(symbol):

    # Specify the function for the weekly data request
    function = 'TIME_SERIES_WEEKLY'
    # Fetch historical stock price data from Alpha Vantage for the symbol
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={API_KEY}"

    response = requests.get(url)
    # Convert data to pandas DataFrame
    data = response.json()["Weekly Time Series"]
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df;


def GetTimeSeriesWeekly(symbol):

    # Specify the function for the weekly data request
    function = 'TIME_SERIES_MONTHLY'
    # Fetch historical stock price data from Alpha Vantage for the symbol
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={API_KEY}"

    response = requests.get(url)
    # Convert data to pandas DataFrame
    data = response.json()["Monthly Time Series"]
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df;

