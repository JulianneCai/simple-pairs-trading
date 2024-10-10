import pandas as pd
import warnings
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

from utils.pairs import PairsFinder


class AutoTrader:
    def __init__(self, price_X, price_Y, symbol_X, symbol_Y, money):
        self.price_X = price_X
        self.price_Y = price_Y

        self.symbol_X = symbol_X
        self.symbol_Y = symbol_Y

        self.money = money
        self.init_balance = money

        self.pair = (symbol_X, symbol_Y)
        
        _, self.hedge_ratio = PairsFinder().get_hedge_ratio(self.pair)

        self.X_pos = 0
        self.Y_pos = 0
        self.pnl = 0

        self.pair = (self.symbol_X, self.symbol_Y)

        _, self.hedge_ratio = PairsFinder().get_hedge_ratio(self.pair)

    def compute_money(self, x_price, y_price):
        """
        Computes the money that we currently have based off our positions

        Params:
            x_price (float): price of stock X
            y_price (float): price of stock Y

        Returns:
            float: money
        """
        return self.X_pos * x_price + self.Y_pos * y_price
    
    def long_spread(self):
        """
        Longs the spread -- that is, we buy X and sell Y
        """
        self.X_pos += 1
        self.Y_pos -= self.hedge_ratio

    def short_spread(self):
        """
        Shorts the spread -- that is, we sell X and buy Y
        """
        self.X_pos -= 1
        self.Y_pos += self.hedge_ratio

    def trade(self, arima_params, train, test):
        """
        Implements a pairs trading strategy. 

        Params:
            arima_params (tuple): ARIMA hyperparameters. Must be of the form (p,q,d),
            where p, q, and d are int
            train (pandas.Series): training dataset
            test (pandas.Series): testing dataset
        
        Returns:
            float: money have left after trading
            pd.DataFrame: dataframe containing information about whether we bought or 
            sold the spread at each time step, and our PnL.
        """
        warnings.filterwarnings(action='ignore')
        window = 60

        spread = pd.concat([train, test], axis=0)

        #  allows us to use rolling average and std data from training set

        spread_ma = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()

        upper_band = spread_ma + 2 * spread_std
        lower_band = spread_ma - 2 * spread_std

        price_X = self.price_X
        price_Y = self.price_Y

        #  ARIMA package likes lists better
        train = train.to_list()

        #  dataframe that keeps track of what the autotrader did
        results = pd.DataFrame(columns=['money', 'buy', 'sell', 'pred'], index=test.index)

        for day in test.index:
            #  fit our ARIMA model
            model = ARIMA(train, order=arima_params)
            model_fit = model.fit()

            results.loc[day, 'money'] = self.money

            #  forecast next day spread using ARIMA 
            next_day_pred = model_fit.forecast()

            results.loc['pred', day] = next_day_pred[0]

            #  short the spread if forecast breaches upper band
            if next_day_pred >= upper_band[day]:
                self.short_spread()
                self.money -= self.compute_money(price_X[day], price_Y[day])
                self.pnl = self.money - self.init_balance

                results.loc[day, 'buy'] = True
                results.loc[day, 'sell'] = False
                results.loc[day, 'pnl'] = self.pnl

                if self.money < 0:
                    print('You are broke')
                    return self.money, results

            #  long the spread if forecast breaches lower band
            elif next_day_pred <= lower_band[day]:
                self.long_spread()
                self.money -= self.compute_money(price_X[day], price_Y[day])
                self.pnl = self.money - self.init_balance

                results.loc[day, 'buy'] = False
                results.loc[day, 'sell'] = True
                results.loc[day, 'pnl'] = self.pnl

                if self.money < 0:
                    print('You are broke')
                    return self.money, results
            #  if the spread is within the bands, just do nothing
            else:
                results.loc[day, 'buy'] = False
                results.loc[day, 'sell'] = False
                results.loc[day, 'pnl'] = self.pnl

            #  append actual spread value to the training set
            #  the next day, we will refit the ARIMA model on this new dataset
            train.append(test[day])

        return self.money, results

    def plot_buy_sell(self, spread, results):
        """
        Plots the spread, and tells us where and when we bought or sold the spread.

        Params:
            spread (pandas.Series): time series data of the spread between stock X and Y
            results (pandas.DataFrame): the results dataframe returned by self.trade()
        """
        sell = spread[(results[results['sell'] == True]).index]
        buy = spread[(results[results['buy'] == True]).index]

        plt.plot(spread.index, spread, label='Spread')
        plt.plot(sell.index, sell, color='r', linestyle='None', marker='^', label='Sell')
        plt.plot(buy.index, buy, color='g', linestyle='None', marker='^', label='Buy')

        plt.xticks(rotation=45, ha='right')
        plt.title('Buy and Sell Signals')
        plt.ylabel('Spread')
        plt.xlabel('Date')

        plt.legend()

        plt.show()

    def plot_buy_sell_stock(self, results):
        """
        Plots where the autotrader bold and sold each stock in the portfolio.

        Params:
            results (pd.DataFrame): the results dataframe returned by self.results()
        """
        _, stock_X = train_test_split(self.price_X, test_size=0.2, shuffle=False)
        _, stock_Y = train_test_split(self.price_Y, test_size=0.2, shuffle=False)

        sell_X = stock_X[(results[results['sell'] == True]).index]
        buy_X = stock_X[(results[results['buy'] == True]).index]

        sell_Y = stock_Y[(results[results['sell'] == True]).index]
        buy_Y = stock_Y[(results[results['buy'] == True]).index]

        plt.plot(stock_X.index, stock_X, label=self.symbol_X, color='black')
        plt.plot(stock_Y.index, stock_Y, label=self.symbol_Y)

        plt.plot(buy_X.index, buy_X, color='g', linestyle='None', marker='^', label='Buy')
        plt.plot(sell_X.index, sell_X, color='r', linestyle='None', marker='^', label='Sell')

        plt.plot(sell_Y.index, sell_Y, color='g', linestyle='None', marker='^')
        plt.plot(buy_Y.index, buy_Y, color='r', linestyle='None', marker='^')

        plt.title(self.symbol_X + ' and ' + self.symbol_Y + ' with Buy and Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')

        plt.legend()
