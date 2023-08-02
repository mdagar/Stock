import yfinance as yf

# Download historical intraday data for NIFTY Microcap 250 Index
data = yf.download('NIFTY_MIDCAP_100.NS', period='1d')  # Replace '^CRSMID' with the correct symbol if it's not correct

print(data)
# Get today's opening price and the most recent price
opening_price = data['Open'].iloc[0]
current_price = data['Close'].iloc[-1]

# Calculate the change
change = current_price - opening_price
# Calculate the percentage change
percentage_change = (change / opening_price) * 100

print(f"The NIFTY Microcap 250 Index has moved by {change} points today, which is a change of {percentage_change}%.")
