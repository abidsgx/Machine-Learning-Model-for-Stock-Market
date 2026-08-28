import pandas as pd
import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
from sklearn.metrics import accuracy_score, log_loss
import joblib

# --- 1. Load and prepare data ---
def load_data():
    df = pd.read_csv('1min_stock_data.csv')

    # Log returns from close prices
    close = df['Close'].to_numpy()
    percentage_change = close[1:] / close[:-1] - 1
    log_returns = np.log1p(percentage_change)
    log_returns = np.clip(log_returns, -0.1, 0.1)

    # Features
    log_series = pd.Series(log_returns)
    open_series = pd.Series(df['Open'].to_numpy())
    high_series = pd.Series(df['High'].to_numpy())
    low_series = pd.Series(df['Low'].to_numpy())
    log_volume = np.log1p(df['Volume'].to_numpy())


    # Feature Engineering
    
    # Lags (past values only)
    lag1 = log_series.shift(1)
    lag2 = log_series.shift(2)
    lag_high = high_series.shift(1)
    lag_low = low_series.shift(1)
    lag_open = open_series.shift(1)

    # Differencing
    diff_high = high_series - lag_high
    diff_low = low_series - lag_low
    diff_open = open_series - lag_open

    # Target = next log return
    target = log_series.shift(-1)

    # Build final DataFrame (only features actually used in X + target)
    df_features = pd.DataFrame({
        'log_series': log_series,
        'log_volume': log_volume,
        'diff_high': diff_high,
        'diff_low': diff_low,
        'lag_1': lag1,
        'lag_2': lag2,
        'diff_open': diff_open,
        'target': target
    })

    df_features.dropna(inplace=True)  # drops first 2 rows, keeps row 2 onward
    return df_features

# --- 2. Extract features and target ---
def extract(df_features=load_data()):
    X = df_features[['log_series', 'log_volume', 'diff_high', 'diff_low',
                     'lag_1', 'lag_2', 'diff_open']].to_numpy()
    y = df_features['target'].to_numpy()
    binary_y = (y > 0).astype(int)

    split = int(0.8 * len(X))
    np.save('split.npy', split)  # save for backtest alignment
    return X, y, binary_y, split

# --- 3. Main training & streaming loop ---
def main():
    patch_sklearn()

    X, y, binary_y, split = extract()

    X_train, X_test = X[:split], X[split:]
    y_train_bin, y_test_bin = binary_y[:split], binary_y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initial model
    model = SGDClassifier(
        loss='modified_huber',
        penalty='elasticnet',
        alpha=0.001,
        learning_rate='adaptive',
        random_state=42
    )
    model.fit(X_train_scaled, y_train_bin)

    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved.")

    print(f"Initial accuracy: {model.score(X_train_scaled, y_train_bin):.4f}")

    # Streaming loop (predict → store → update)
    window_size = 118
    half_life = 400

    x_buffer = deque(maxlen=window_size)
    y_buffer = deque(maxlen=window_size)

    predictions = []
    true_vals = []
    probabilities = []

    for i in range(len(X_test_scaled)):
        x_point = X_test_scaled[i].reshape(1, -1)
        y_true = y_test_bin[i]

        # 1. Predict (before updating)
        pred = model.predict(x_point)[0]
        prob = model.predict_proba(x_point)[0][1]

        predictions.append(pred)
        true_vals.append(y_true)
        probabilities.append(prob)

        # 2. Buffer the point
        x_buffer.append(X_test_scaled[i])
        y_buffer.append(y_true)

        # 3. Update model when buffer is full (NO RELOADING!)
        if len(x_buffer) == window_size:
            X_batch = np.array(x_buffer)
            y_batch = np.array(y_buffer)

            time_diffs = np.arange(window_size - 1, -1, -1)
            weights = np.exp(-(np.log(2) * time_diffs) / half_life)

            # Incrementally update the model in memory
            model.partial_fit(X_batch, y_batch, sample_weight=weights, classes=[0, 1])

    # Save the final updated model
    joblib.dump(model, 'model_updated.pkl')


    # Evaluation
    acc = accuracy_score(true_vals, predictions)
    logloss = log_loss(true_vals, probabilities)

    print(f"Streaming Accuracy: {acc:.4f}")
    print(f"Streaming Log-Loss: {logloss:.4f}")

    high_conf = np.sum(np.array(probabilities) > 0.65)
    print(f"Number of predictions with confidence > 0.65: {high_conf}")

    # Save metrics for backtesting
    metrics = {
        'Predictions': predictions,
        'True Values': true_vals,
        'Probabilities': probabilities
    }
    pd.DataFrame(metrics).to_csv('metrics.csv', index=False)

if __name__ == '__main__':
    main()
