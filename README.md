# Pairs Trading using ARIMA and XGBoost

Please see the Jupyter notebook to see how to use the code. 

The utils file contains a lot of useful classes with useful methods for our data analysis.

`pairs.py` contains the `PairFinder` class, which takes in a list of stock symbols as input. It then looks up their close prices using the `yfinance` package, and then identifies cointegrated pairs using the `coint()` method from `statsmodels`. It also has methods for plotting the spread, and computing the hedge ratio of the two stocks.

`trainer.py` contains methods for fitting an ARIMA model

`tuner.py` contains methods for tuning the hyperparameters of an ARIMA model. It performs a grid search on a list of p, q, and d values, and then selects the best ones best of either AIC or BIC.

`trader.py` contains the autotrader bot that implements our pairs trading strategy. It also has methods that plots buy-sell signals of the spread, and also the times at which the autotrader buys and sells each stock.
![Untitled](https://github.com/user-attachments/assets/d4a3be68-fd33-4fdf-b83a-5aad13d168a3)
![Untitled](https://github.com/user-attachments/assets/ba592840-9bd2-41fa-808c-33a693188c47)



# Goal of this Project
 1. Identified suitable pairs of stocks by testing each one for cointegration
 2. Determined hedge ratio using linear regression
 3. Used moving average and constructed two-standard-deviation Bollinger bands
 4. Use ARIMA model to forecast spread for the next day
 5. If forecasted spread breaches upper band, short spread. If forecasted spread breaches lower band, long spread.
 6. Profit?

# Data 
All data for this project was obtained from the `yfinance` package. 

# Python Packages Used
 - pandas
 - statsmodels
 - matplotlib
 - scikit-learn
 - seaborn
 - numpy
 - yfinance
