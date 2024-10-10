import pandas as pd
import yfinance as yf
import numpy as np
from numpy import sqrt, log

from sklearn.metrics import root_mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import adfuller, coint


class Trainer:
    """
    Abstract class representing an object that trains a machine learning estimator 
    on a training dataset, and then uses that to predict features, either in-sample 
    or out-of-sample.
    """
    def __init__(self, symbol, period):
        """
        Params:
            symbol (str): the stock symbol (e.g. "GOOG" for Google, "AAPL" for apple)

            period (str): time period of stocks -- must be one of 
            {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}

            test_size (float): size of test dataset. Sum of test_size and train_size must equal 1

            train_size (float): size of training dataset. Sum of test_size and train_size must equal 1
        """
        self._symbol = symbol
        self._period = period

    def generate_features(self):
        """
        Generates all the features that we need for predicting close prices of stocks. 
        We focus only on features that would be available to us at open to prevent data leakage.

        Returns:
            pandas.DataFrame: dataframe with all the new features
        """
        df = yf.Ticker(self._symbol).history(period=self._period)

        #  we are using open price to predict close prices.
        #  these variables will not be known to us at open, so they are being 
        #  dropped to avoid data leakage
        df.drop(columns=['High', 'Low'], axis=1, inplace=True)

        df['daily_returns'] = df['Open'].pct_change()

        df['daily_returns_squared'] = df['daily_returns'] ** 2

        window = 2

        df['hist_vol'] = df['daily_returns'].rolling(window=window).std() * sqrt(252 / window)

        df['log_returns'] = log(df['Open']).diff()

        df['real_vol'] = df['log_returns'].rolling(window=window).std() * sqrt(252 / window)

        df['volume_change'] = df['Volume'].diff()

        df.index = df.index
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month 
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
         
        return df
    
    def generate_out_of_sample_features(self, df, lags):
        """
        For out-of-sample predictions. In this case, the only features that 
        we will have access to are the lags, and the dates.

        Returns:
            pandas.DataFrame: dataset with lag and date features
        """
        df = pd.DataFrame(df)
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = pd.to_datetime(df.index).quarter
        df['month'] = df.index.month 
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        
        return df
 
    def is_stationary(self, df):
        """
        Checks if the time series is stationary using the Augmented Dickey-Fuller test

        :params df: time series data 
        :type df: pandas.DataFrame 

        :returns: (bool) true if stationary, false if not
        """
        p_value = adfuller(df)[1]
        print(f'p-value of ADF test: {p_value}')
        if p_value <= 0.05:
            return True 
        else:
            return False
        
    def walk_forward_eval(self):
        """
        Abstract method implemented by subclasses
        """
        pass

 
    def get_lags(self, df):
        """
        Obtains the statistically significant autocorrelation lags

        Returns:
            list of int: statistically siginificant lag values
        """
        selection_results = ar_select_order(df, maxlag=8)
        return selection_results.ar_lags

    def get_symbol(self):
        """
        Returns:
            str: stock symbol
        """
        return self._symbol
    
    def set_symbol(self, symbol):
        """
        Params:
            symbol (str): new stock symbol
        """
        self._symbol = symbol 

    def get_period(self):
        """
        Returns:
            str: time period
        """
        return self._period 
    
    def set_period(self, period):
        """
        Params:
            period (str): new time period
        """
        self._period = period


class XGBoostTrainer(Trainer):
    def __init__(self, symbol, period):
        super().__init__(symbol, period)

    def forecast_in_sample(self, estimator, x_train, y_train, x_test, y_test):
        model = estimator.fit(x_train, y_train, 
                              eval_set = [(x_train, y_train), (x_test, y_test)]
                              )
        y_pred = pd.DataFrame(model.predict(x_test), index=y_test.index)
        return y_pred



class ARIMATrainer(Trainer):
    """
    Concrete class representing an object that trains an ARIMA model, and then 
    predicts future values of target feature.
    """
    def __init__(self, symbol, period):
        super().__init__(symbol, period)
    
    def walk_forward_eval(self, params, y_train, y_test):
        """
        Performs walk-forward analysis on the testing dataset. Model trains itself on the 
        training dataset, and makes a prediction for the next time step. The true value of 
        the next time step is appended to the training dataset, and the model is re-fitted using 
        the same parameters.

        Params:
            y_train (pandas.Series): the time series training dataset 
            y_test (pandas.Series): the time series testing dataset
        Returns:
            pandas.Series: predictions made by the ARIMA model
        """
        y_preds = []
        train = y_train.to_list()
        test = y_test.to_list()
        for obs in test:
            model = ARIMA(train, order=params)
            model_fit = model.fit()
            pred = model_fit.forecast()
            y_preds.append(pred)
            train.append(obs)

        print(root_mean_squared_error(y_preds, test))
        
        return y_preds
