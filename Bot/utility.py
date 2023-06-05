import json
import os
import pandas as pd
import yfinance as yf
import time
import csv
import json

def get_scan_type():
    index =""
    scantype=""
    with open('scan.json') as file:
      scantype_json = json.load(file)
      enabled_data = [item for item in scantype_json['ScanType'] if item['IsEnabled']]
      if enabled_data:
        first_enabled_item = enabled_data[0]  # get the first enabled item
        index = first_enabled_item['Index']
        scantype = first_enabled_item['bucket']
    return index ,scantype

def get_stock_list(filename): 
    with open(filename) as file:
      stocklist = json.load(file)
      return stocklist

def load_balance(balance_filename,current_balance):
    if os.path.exists(balance_filename):
        with open(balance_filename, 'r') as file:
            current_balance = float(file.read())
    return current_balance

def load_portfolio(portfolio_filepath):
    portfolio={}
    if os.path.exists(portfolio_filepath):
        portfolio_df = pd.read_csv(portfolio_filepath, index_col=0)
        portfolio = portfolio_df.to_dict(orient='index')
    else:
        portfolio={}
    return portfolio

def get_variable(variablename):
    with open('scan.json') as file:
      parsed_data = json.load(file)
      variable = parsed_data['Variables'][0][variablename]
    return variable
  
def calculate_balance(portfolio, current_balance):
    total_value = 0
    for symbol in portfolio.keys():
        #print(symbol)
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
        total_value += portfolio[symbol]['Quantity'] * current_price
    print(f"Cash: {current_balance} Stock {total_value} Total Balance is: {total_value+current_balance}")

def scan_and_sleep(isscanning, stocknumber, totalstocks,sleep_time=60):
    if(isscanning):
        print(f"Scanning stock {stocknumber}/{totalstocks}", end ='\r')
    else:
        for remaining in range(sleep_time, 0, -5):
            print(f"Next Scan in .. {remaining}s ", end='\r')
            time.sleep(5)




def convertJson():
    csv_file = 'niftymicrocap250_list.csv'
    json_file = 'output.json'
    data = []

    with open(csv_file, 'r') as file:
        csv_data = csv.DictReader(file)
        for row in csv_data:
            row['Symbol'] = row['Symbol'] + '.ns'
            data.append(row)

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

    print("Conversion completed. JSON file created: ", json_file)

