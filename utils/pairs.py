import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint

from sklearn.linear_model import LinearRegression


class PairsFinder:
    """
    Class for finding pairs for pairs trading strategy.
    """
    def __init__(self):
        pass

    def _find_pairs(self, stocks):
        """
        Finds pairs that are cointegrated.

        Params:
            stocks (list<str>): list of stock symbols

        Returns:
            np.array: matrix of t-statistic of unit-root test on residuals
            np.array: matrix of p-values 
            list<tuple>: list of tuples of pairs of stock symbols
        """
        # initialise dataframe, with columns given by stock symbols
        prices = pd.DataFrame(columns=stocks)

        for stock in stocks:
            #  access price data using yfinance package
            data = yf.Ticker(stock).history(period='5y')
            #  take log prices of stocks, and put them into dataframe
            prices[stock] = np.log(data['Close'])
        # number of stocks
        n = int(prices.shape[1])
        #  empty matrix
        score_matrix = np.zeros((n, n))
        # empty pvalue matrix
        pvalue_matrix = np.ones((n, n))

        keys = prices.keys()
        #  list for storing pairs of cointegrated stocks
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                # take each log stock price
                S1 = prices[keys[i]]
                S2 = prices[keys[j]]

                #  test for cointegration
                result = coint(S1, S2)
                #  score
                score = result[0]
                #  pvalue of cointegration test
                pvalue = result[1]
                
                #  populate matrices
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                #  if cointegration test is statistically significant
                #  append to pairs
                if pvalue < 0.05:
                    pairs.append((keys[i], keys[j]))

        return score_matrix, pvalue_matrix, pairs
    
    def plot_coint_heatmap(self, stocks):
        """
        Plots heatmap of p-values of cointegration tests

        Params:
            stocks (list<str>): list of stock symbols

        Returns:
            list<tuple>: list of pairs identified by cointegration test
        """
        _, pvalues, pairs = self._find_pairs(stocks)
        sns.heatmap(
            pvalues, 
            xticklabels=stocks, 
            yticklabels=stocks, 
            cmap='RdYlGn_r',
            mask=(pvalues>=0.98),
            annot=True
        )

        return pairs
    
    def get_spread(self, pair):
        """
        Calculates hedge ratio of pair of stocks, and then computes the spread.

        Params:
            pair (tuple): tuple of stock symbols

        Returns:
            pandas.Series: the spread of the log price between the two stocks
        """
        #  take log price of stocks
        log_price_X = np.log(yf.Ticker(pair[0]).history(period='5y')['Close'])
        log_price_Y = np.log(yf.Ticker(pair[1]).history(period='5y')['Close'])

        _, hedge_ratio = self.get_hedge_ratio(pair)

        spread = log_price_X - hedge_ratio * log_price_Y

        return spread
    
    def plot_spread(self, pair):
        """
        Plots the spread of a pair of stocks

        Params:
            pair (tuple): tuple of stock symbols
        """
        spread = self.get_spread(pair)

        plt.plot(spread.index, spread, label=pair)
        plt.xlabel('Date')
        plt.ylabel('Log Price Difference')

        plt.legend()

    def get_hedge_ratio(self, pair):
        """
        Obtains the hedge ratio of a pair of stocks.

        Params:
            pair (tuple): tuple of stock symbols
        
        Returns:
            list: predictions made by linear regression model
            float: hedge ratio given by slope of linear regression
        """
        log_price_X = np.log(yf.Ticker(pair[0]).history(period='5y')['Close']).values.reshape(-1, 1)
        log_price_Y = np.log(yf.Ticker(pair[1]).history(period='5y')['Close']).values.reshape(-1, 1)

        lin_model = LinearRegression(fit_intercept=True).fit(
            log_price_X,
            log_price_Y
        )

        #  we just want to get a best fit line, so we use it to predict itself
        pred = lin_model.predict(log_price_X)

        try: 
            #  coef_ attribute occasional returns double list [[coef_]]
            #  this happens if fit_intercept option is set to True
            hedge_ratio = lin_model.coef_[0][0]
        except TypeError:
            #  in case fit_intercept option is False, just access first element of list
            #  for the gradient of the line
            hedge_ratio = lin_model.coef_[0]

        #  rounds up the hedge ratio to an integer
        #  it doesn't make sense to buy half a stock
        dec = abs(np.floor(hedge_ratio) - hedge_ratio)
        if dec >= 0.5:
            hedge_ratio = np.ceil(hedge_ratio)
        else:
            hedge_ratio = np.floor(hedge_ratio)

        return pred, hedge_ratio

    def plot_hedge_ratio(self, pair):
        """
        Gives a scatter plot of the pair of stocks, along with the best fit line
        between them.

        Params:
            pair (tuple): tuple of stock symbols

        Returns:
            float: the hedge ratio
        """
        log_price_X = np.log(yf.Ticker(pair[0]).history(period='5y')['Close']).values.reshape(-1, 1)
        log_price_Y = np.log(yf.Ticker(pair[1]).history(period='5y')['Close']).values.reshape(-1, 1)

        pred, hedge_ratio = self.get_hedge_ratio(pair)

        plt.plot(
            log_price_X, 
            pred,
            color='r'
        )

        plt.scatter(log_price_X,
                    log_price_Y,
                    s=3, 
                    label=pair[0] + ' vs. ' + pair[1])

        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.legend()

        return hedge_ratio
