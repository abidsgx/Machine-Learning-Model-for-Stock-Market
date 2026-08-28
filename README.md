# Machine-Learning-Models-for-Stock-Market

<img width="1350" height="583" alt="newplot" src="https://github.com/user-attachments/assets/aaf23e12-6947-40b2-be83-61093d5b19ec" />

Do Not Use These for Trading!

This repository contains a classification model (SGDClassifier) that can be trained and evaluated on time series stock data. It also includes a backtesting script that will allow you to visualise the models performance.

## Data Format
The stock data should be in a csv format and should have the Open, High, Low, Close, and Volume columns.

<img width="416" height="106" alt="image" src="https://github.com/user-attachments/assets/51bd0342-8a80-453b-b077-bddd7a4d8a72" />

If the naming is different, then the script needs to be edited. If your data does not include volume, then it should be removed in the script as well. It is fine if other columns are present at first.

Edit these lines so that the column names mach the ones in your csv file:

```
close = df['Close'].to_numpy()
...
log_series = pd.Series(log_returns)
open_series = pd.Series(df['Open'].to_numpy())
high_series = pd.Series(df['High'].to_numpy())
low_series = pd.Series(df['Low'].to_numpy())
log_volume = np.log1p(df['Volume'].to_numpy())
```

## Setup
The price classification model simply predicts whether the price will close higher or lower in the next minute based on the returns of the last minute and other features such as lagged returns and differencing. Since the model uses data from the last minutes close price, it may be hard to deploy it in real time, so feel free to change the features to make it more realistic.

To use the model, download the script and move it into the same folder that contains your stock data. Make sure the data is in csv format. Then edit the script so that 

`csv = 'name of your data file.csv'`

Also edit the column names in the script (Open, High, Low, Close, Volume, etc.) so that they match the column names in your file.

Install the required libraries using this command in the terminal:

`pip install pandas numpy scikit-learn scikit-learn-intelex joblib plotly numba`

Open the terminal in your folder then run:

`python price_classification_model.py`

To run the backtesting script, download it and move it into the same folder that contains your stock data and classification model. You need to run the classification model at least once so that it saves the model and metrics in the folder for the backtesting script to work.

Then edit the script and set csv = '' to the same file you used for the model. Then go to line 91 and set 'Close' to whatever the close price column name is in your data file.

