import pandas as pd
import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearnex import patch_sklearn
from sklearn.metrics import accuracy_score, log_loss
import joblib
import matplotlib.pyplot as plt

# Write the name of your .csv file containing the time series data (include .csv)

csv = ''

# --- 1. Load and prepare data ---
def load_data():
    df = pd.read_csv(csv)

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

df_features = load_data()


# --- 2. Extract features and target ---
def extract(features=df_features):
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
        loss='log_loss',
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

    train_sizes = []
    train_scores = []
    test_scores = []

    current_size = 0

    loss_history = []
    iteration_counter = 0
    iterations = []

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

            # Track metrics
            current_size += len(X_batch)
            train_sizes.append(current_size)

            # Score on current seen data vs static validation/test set
            train_scores.append([model.score(X_batch, y_batch)])
            test_scores.append([model.score(X_test_scaled, y_test_bin)])

            # Calculate loss over the entire dataset (or validation set)
            # Note: clf.predict_proba is available only for 'log_loss' and 'modified_huber'
            y_prob = model.predict_proba(X_test_scaled)
            current_loss = log_loss(y_test_bin, y_prob)

            # Track historical benchmarks
            loss_history.append(current_loss)
            iterations.append(iteration_counter)
            iteration_counter += 1

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

    # --- Prepare data for plots ---
    coefficients = model.coef_.ravel()
    feature_names = df_features.columns[:-1]  # all except 'target'

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test_bin, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])

    # 2. Learning Curve
    train_sizes = np.array(train_sizes)
    train_scores = np.array(train_scores)  # shape: (n_updates, 1)
    test_scores = np.array(test_scores)  # shape: (n_updates, 1)

    # --- Add 2x2 subplot grid ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))  # Larger figure size

    fig.canvas.manager.set_window_title('Model Performance')

    # --- Padding between subplots ---
    plt.subplots_adjust(
        left=0.08,  # left margin
        right=0.09,  # right margin
        bottom=0.08,  # bottom margin
        top=0.09,  # top margin
        wspace=0.5,  # width space between subplots
        hspace=0.5  # height space between subplots
    )

    # Subplot (1,1): Confusion Matrix
    disp.plot(ax=axes[0, 0], cmap=plt.cm.Blues)
    axes[0, 0].set_title('Confusion Matrix (Test Set)', fontsize=8)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=8)
    axes[0, 0].set_ylabel('True Label', fontsize=8)

    # Subplot (1,2): Incremental Learning Curve
    display = LearningCurveDisplay(
        train_sizes=train_sizes,
        train_scores=train_scores,
        test_scores=test_scores,
        score_name="Accuracy"
    )
    display.plot(ax=axes[0, 1])
    axes[0, 1].set_title('Incremental Learning Curve', fontsize=8)
    axes[0, 1].set_xlabel('Number of Samples Processed', fontsize=8)
    axes[0, 1].set_ylabel('Accuracy', fontsize=8)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend(fontsize=11)

    # Subplot (2,1): Feature Weights
    bars = axes[1, 0].bar(feature_names, coefficients, color='steelblue', alpha=0.8)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 0].set_xlabel('Feature', fontsize=8)
    axes[1, 0].set_ylabel('Coefficient Value', fontsize=8)
    axes[1, 0].set_title('Feature Weights (Final Model)', fontsize=8)

    # Rotate x-axis labels 45 degrees to prevent overlap
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=10)
    # Add value labels on bars
    for bar, val in zip(bars, coefficients):
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.001 * np.sign(height) if height != 0 else 0.001,
            f'{val:.3f}',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=8,
            rotation=0
        )

    # Subplot (2,2): Loss Curve
    axes[1, 1].plot(iterations, loss_history, label='Validation Log-Loss',
                    color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    axes[1, 1].set_title('SGDClassifier Loss Curve', fontsize=8)
    axes[1, 1].set_xlabel('Batch Iterations', fontsize=8)
    axes[1, 1].set_ylabel('Log Loss', fontsize=8)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend(fontsize=11)

    # Add min/max annotations on loss curve
    if loss_history:
        min_loss_idx = np.argmin(loss_history)
        min_loss_val = loss_history[min_loss_idx]
        axes[1, 1].annotate(f'Min: {min_loss_val:.4f}',
                            xy=(iterations[min_loss_idx], min_loss_val),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=10,
                            color='red',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 

    plt.show()

if __name__ == '__main__':
    main()
