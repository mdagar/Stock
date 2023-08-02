import os
import pandas as pd
import yfinance as yf
import talib as ta
import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class StockPredictor:
    def __init__(self, tickers, start_date=pd.Timestamp.today() - pd.DateOffset(years=5), end_date=pd.Timestamp.today()):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}

    def fetch_data(self):
        self.data = {}
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, interval='1d')
            df['RSI'] = ta.RSI(df['Close'])
            macd, macdsignal, macdhist = ta.MACD(df['Close'])
            df['MACD'] = macd
            df['MACD_SIGNAL'] = macdsignal
            df['MACD_HIST'] = macdhist
            df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'])
            df['ADXR'] = ta.ADXR(df['High'], df['Low'], df['Close'])
            df['+DI'] = ta.PLUS_DI(df['High'], df['Low'], df['Close'])
            df['-DI'] = ta.MINUS_DI(df['High'], df['Low'], df['Close'])
            df['target'] = df['Close'].pct_change().shift(-1)
            df = df.dropna()
            self.data[ticker] = df

    def create_model(self):
        for ticker in self.tickers:
            if not os.path.exists(f'{ticker}_model.h5'):
                df = self.data[ticker]
                X = df[['RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ADX', 'ADXR', '+DI', '-DI']].values
                y = df['target'].values
                X = self.scaler.fit_transform(X)

                def build_model(hp):
                    model = keras.Sequential()
                    for i in range(hp.Int('num_layers', 2, 20)):
                        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                                                    activation='relu'))
                    model.add(keras.layers.Dense(1, activation='linear'))
                    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                                loss='mean_absolute_error',
                                metrics=['mean_absolute_error'])
                    return model

                tuner = RandomSearch(build_model, objective='val_mean_absolute_error', max_trials=5, executions_per_trial=3)
                tuner.search(X, y, epochs=100, validation_split=0.2)
                self.models[ticker] = tuner.get_best_models(num_models=1)[0]
                self.models[ticker].fit(X, y, epochs=100, validation_split=0.2)
                self.models[ticker].save(f'{ticker}_model.h5')

    def predict(self, ticker):
        if os.path.exists('general_model.h5'):
            model_general = keras.models.load_model('general_model.h5')
        else:
            print("General model does not exist. Please train the model.")
            #return
        if os.path.exists(f'{ticker}_model.h5'):
            model_stock = keras.models.load_model(f'{ticker}_model.h5')
        else:
            print(f"Model for {ticker} does not exist. Please train the model.")
            return

        start_date = pd.Timestamp.today() - pd.DateOffset(months=6)
        end_date = pd.Timestamp.today()
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        df['RSI'] = ta.RSI(df['Close'])
        macd, macdsignal, macdhist = ta.MACD(df['Close'])
        df['MACD'] = macd
        df['MACD_SIGNAL'] = macdsignal
        df['MACD_HIST'] = macdhist
        df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'])
        df['ADXR'] = ta.ADXR(df['High'], df['Low'], df['Close'])
        df['+DI'] = ta.PLUS_DI(df['High'], df['Low'], df['Close'])
        df['-DI'] = ta.MINUS_DI(df['High'], df['Low'], df['Close'])
        #df = df.dropna()
        print(df)
        X = df[['RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ADX', 'ADXR', '+DI', '-DI']]
        
        # prediction_general = model_general.predict(X)[-1][0]
        # print(f"Predicted % change for {ticker} using general model: {prediction_general * 100}%")
        # predicted_price_general = df['Close'].values[-1] * (1 + prediction_general)
        # print(f"Predicted price for {ticker} in 15 mins using general model: {predicted_price_general}")

        print(X)
        prediction_stock = model_stock.predict(X)[-1][0]

        print(f"Predicted % change for {ticker} using stock-specific model: {prediction_stock * 100}%")
        predicted_price_stock = df['Close'].values[-1] * (1 + prediction_stock)
        print(f"Predicted price for {ticker} in 15 mins using stock-specific model: {predicted_price_stock}")

        return 0, predicted_price_stock

    def analyze(self, filename):
        df = pd.read_csv(filename)
        plt.plot(df['Time'], df['Actual price'], label='Actual price')
        plt.plot(df['Time'], df['Predicted price_general'], label='Predicted price (general model)')
        plt.plot(df['Time'], df['Predicted price_stock'], label='Predicted price (stock-specific model)')
        plt.legend()
        plt.show()


stock_predictor = StockPredictor(['AAPL', 'GOOGL', 'MSFT'])
stock_predictor.fetch_data()
stock_predictor.create_model()
predicted_price_general, predicted_price_stock = stock_predictor.predict('AAPL')
#stock_predictor.analyze('prediction_results.csv')
