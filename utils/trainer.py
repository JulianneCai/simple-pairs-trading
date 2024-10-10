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
