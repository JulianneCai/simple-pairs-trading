import pandas as pd
import matplotlib.pyplot as plt

from numpy import sqrt, log

from sklearn.metrics import root_mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.stattools import adfuller


class Trainer:
    """
    Abstract class representing an object that trains a machine learning estimator 
    on a training dataset, and then uses that to predict features, either in-sample 
    or out-of-sample.
    """
    def __init__(self):
        pass

    def generate_features(self, df):
        """
        Generates all the features that we need

        Returns:
            pandas.DataFrame: dataframe with all the new features
        """

        df['daily_returns_close'] = df['Close'].pct_change()

        df['daily_returns_close_squared'] = df['daily_returns_close'] ** 2

        window = 20

        df['volume_change'] = df['Volume'].diff()

        df['hist_vol_close'] = df['daily_returns_close'].rolling(window=window).std() * sqrt(252 / window)
      
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
            print('The time series is stationary')
            return True 
        else:
            print('The time series is not stationary')
            return False
        
    def plot_importance(self, estimator, n=0):
        """
        Plots the n most important features using the gini criterion.

        Params:
            (int) n: the number of important features to be plotted
        """
        feature_names = estimator.feature_names_in_
        feature_importances = estimator.feature_importances_

        importance_dict = {key: value for key, value in zip(feature_names, feature_importances)}

        sorted_importances = dict(sorted(importance_dict.items(), key=lambda item: item[1]))
        sorted_importances = dict(list(sorted_importances.items())[-n:])

        plt.barh(sorted_importances.keys(), sorted_importances.values())

    
    def generate_out_of_sample_features(self, df, lags, target):
        """
        For out-of-sample predictions. In this case, the only features that 
        we will have access to are the lags, and the dates.
        Returns:
            pandas.DataFrame: dataset with lag and date features
        """
        df = pd.DataFrame(df)
        df.index = pd.to_datetime(df.index)
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = pd.to_datetime(df.index).quarter
        df['month'] = df.index.month 
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day

        for lag in lags:
            df['lag_' + str(lag)] = df[target].diff(lag)

        return df
        
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



class ARIMATrainer(Trainer):
    """
    Concrete class representing an object that trains an ARIMA model, and then 
    predicts future values of target feature.
    """
    def __init__(self):
        super().__init__()
    
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
        #  convert training and testing dataset to list
        train = y_train.to_list()
        test = y_test.to_list()
        for obs in test:
            #  re-train ARIMA model with new training dataset on each day
            model = ARIMA(train, order=params)
            #  fit model to training data
            model_fit = model.fit()

            #  forecast one day ahead 
            pred = model_fit.forecast()
            y_preds.append(pred[0])
            train.append(obs)

        print(f'RMSE: {root_mean_squared_error(y_preds, test)}')
        
        return y_preds
