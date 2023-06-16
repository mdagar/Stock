import json
import os
import pandas as pd
import yfinance as yf
import time
import csv
import json
import multiprocessing

def get_enabled_scan_types():
        with open('scan.json') as file:
            data = json.load(file)
            if "ScanType" in data:
                scan_types = data["ScanType"]
                enabled_scan_types = [scan for scan in scan_types if scan.get("IsEnabled", True)]
                return enabled_scan_types

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
  

def scan_and_sleep(isscanning, stocknumber, totalstocks,sleep_time=60): 
    process_name = multiprocessing.current_process().name
    if(isscanning):
        print(f"{process_name}-(Scanning {stocknumber}/{totalstocks}) --", end ='\r')
    else:
        for remaining in range(sleep_time, 0, -5):
            print(f"{process_name} Next Scan in .. {remaining}s ", end='\r')
            time.sleep(5)

def convertJson():
    csv_file = 'ind_niftymidcap100list.csv'
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

#convertJson()

def save_trade(trades,tradefile):
    # Save trade information to CSV with today's date appended to the filename
    if os.path.exists(tradefile):
        existing_trades = pd.read_csv(tradefile)
        all_trades = pd.concat([existing_trades, trades], ignore_index=True)
        all_trades.to_csv(tradefile, index=False)
    else:
        trades.to_csv(tradefile,index=False)     

def save_balance(current_balance,balance_filename):
    # Save the updated balance to the file
    with open(balance_filename, 'w') as file:
       file.write(str(current_balance))

