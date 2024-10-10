from statsmodels.tsa.arima.model import ARIMA



class ARIMATuner():
    """
    Concrete class representing an object that tunes ARIMA hyperparameters:
    parameters p and q, and the unit root term d
    """
    def __init__(self):
        super().__init__()

    def _aic_arima_model(self, p, q, d, train):
        """
        Calculates the Akaike Information Criterion (AIC) of an ARIMA(p,q,d) model

        Returns:
            float: the AIC of an ARIMA(p,q,d) model
        """
        cfg = (p, q, d)
        model = ARIMA(train, order=cfg)
        model_fit = model.fit()

        return model_fit.aic

    def _bic_arima_model(self, p, q, d, train):
        """
        Calculates the Bayesian Information Criterion (BIC) of an ARIMA(p,q,d) model.

        Returns:
            float: the BIC of an ARIMA(p,q,d) model
        """
        cfg = (p, q, d)
        model = ARIMA(train, order=cfg)
        model_fit = model.fit()

        return model_fit.bic
    
    def _find_d_values(self, train):
        """
        Finds order of differencing for time series data.

        Returns: 
            list<int>: orders of differencing
        """
        d_values = []
        if self.is_stationary(train):
            return 0
        else:
            #  start with first order difference of time series
            i = 1
            #  continue taking higher order differences until stationary
            while self.is_stationary(self._y_train.shift(i)) is False:
                d_values.append(i)
                i += 1
        return d_values

    def grid_search(self, param_space, method, train):
        """
        Tunes ARIMA hyperparameters by performing a grid search, using 
        Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC)
        as the scoring function.

        Params:
            dict<str: list<int>>: dictionary object with keys given by strings, 
            which must be labelled 'p', 'q', and 'd', with values given by 
            lists of integers. I.e. {'p': [1,2,3,4], 'q': [1,2,3,4], 'd': [1,2]}
        
        Returns:
            tuple: the hyperparmeters p, q, and d
        """
        best_score = float('inf') 
        score = None
        best_cfg = None

        #  the p and q hyperparameters are given by statistically significant 
        #  lags on the autocorrelation plot
        p_values = param_space['p']
        q_values = param_space['q']
        d_values = param_space['d']

        for p in p_values:
            for q in q_values:
                for d in d_values:
                    cfg = (p, q, d)
                    if method == 'aic':
                        score = self._aic_arima_model(p, q, d, train)
                    elif method == 'bic':
                        score = self._bic_arima_model(p, q, d, train)
                    print(f'{method.upper()} score = {score}, cfg = {cfg}')
                    if score < best_score:
                        best_score = score
                        best_cfg = cfg 
        return best_cfg
