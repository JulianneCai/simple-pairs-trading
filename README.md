# Pairs Trading with ARIMA

Please see the Jupyter notebook to see how to use the code. 

The utils file contains a lot of useful classes with useful methods for our data analysis.

`pairs.py` contains the `PairFinder` class, which takes in a list of stock symbols as input. It then looks up their close prices using the `yfinance` package, and then identifies cointegrated pairs using the `coint()` method from `statsmodels`. It also has methods for plotting the spread, and computing the hedge ratio of the two stocks.

`trainer.py` contains methods for fitting an ARIMA model

`tuner.py` contains methods for tuning the hyperparameters of an ARIMA model. It performs a grid search on a list of p, q, and d values, and then selects the best ones best of either AIC or BIC.

`trader.py` contains the autotrader bot that implements our pairs trading strategy. It also has methods that plots buy-sell signals of the spread, and also the times at which the autotrader buys and sells each stock.
![image](https://github.com/user-attachments/assets/457c2be3-da25-4dd6-9da4-ebe2894dcd7c)
