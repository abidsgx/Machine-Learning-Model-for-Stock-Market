import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from numba import jit, boolean
from sklearn.metrics import accuracy_score, log_loss

# Write the name of your csv file here that was used for the classification model (include .csv)

csv = ''

# --- 1. EVALUATE on streaming predictions ---

df = pd.read_csv('metrics.csv')

predictions, true_vals, probabilities = df['Predictions'].to_numpy(), df['True Values'].to_numpy(), df['Probabilities'].to_numpy()

acc = accuracy_score(true_vals, predictions)
logloss = log_loss(true_vals, probabilities)

print(f"Streaming Accuracy: {acc:.4f}")
print(f"Streaming Log-Loss: {logloss:.4f}")

# Count how many predictions had confidence > 0.57 (using streaming probabilities)
high_conf_count = np.sum(np.array(probabilities) > 0.65)
print(f"Number of predictions with confidence > 0.65: {high_conf_count}")

# Plotting

rmse = mean_squared_error(true_vals, predictions)

p = np.array(predictions)   # shape (N,)
t = np.array(true_vals)   # shape (N,)

p_up = p == 1
p_down = p == 0
actual_up = t == 1
actual_down = t == 0

# Correct if both predict and actual go same direction from current price
correct_dir = (p_up & actual_up) | (p_down & actual_down)

# Prepend False for first point (no previous to compare)
highlight = correct_dir

# Now highlight[i] is True if the model correctly predicted the direction at time i (i>0)

# Assume t and p are 1D numpy arrays of same length N
# highlight is already computed (length N) as defined above

# Actual changes and their signs                  # length N-1

actual_sign = np.where(t == 1, 1, 0)

# Reversal: sign flips between consecutive steps
reversal = (actual_sign[:-1] != actual_sign[1:])

# Correct prediction for the step after reversal: we need highlight[r+2] for each reversal r
# highlight[2:] has length N-2 and aligns with reversal
correct_on_reversal = reversal & highlight[:-1]

# Accuracy (avoid division by zero if no reversals)
if reversal.sum() > 0:
    reversal_accuracy = correct_on_reversal.sum() / reversal.sum() * 100
else:
    reversal_accuracy = np.nan

true_predictions = np.where(highlight == True, 1, 0)

accuracy = np.mean(highlight) * 100

recent_start = int(0.9 * len(true_predictions))
recent_accuracy = np.mean(true_predictions[recent_start:]) * 100

recent_reversal = reversal[recent_start:]
recent_correct = correct_on_reversal[recent_start:]

if recent_reversal.sum() > 0:
    recent_reversal_accuracy = recent_correct.sum() / recent_reversal.sum() * 100
else:
    recent_reversal_accuracy = np.nan

# Backtesting

split = np.load('split.npy')

# Load original data

df_stock = pd.read_csv(csv)

close = df_stock['Close'].to_numpy()

returns_decimal = (close[1:] - close[:-1])/close[1:]
forward_returns = pd.Series(returns_decimal[split:]).shift(-1)  # return from t to t+1

close_prices = close[split:]


@jit(nopython=True)
def run_backtest(p_up, confidence_levels, forward_returns, closes,
                 initial_cash=1000.0, entry_cost=0.00, exit_cost=0.00):
    """
    Runs the backtest. All arrays must be 1D numpy arrays of the same length.
    """
    # State variables
    position = False
    shares_held = 0
    cash_held = initial_cash
    market_value = 0.0

    # Portfolio tracking
    n = len(p_up)
    portfolio = np.zeros(n + 1)  # Pre-allocate for speed
    portfolio[0] = initial_cash

    # Loop (compiled by Numba -> runs almost as fast as C)
    for i in range(n):
        up = p_up[i]
        ret = forward_returns[i]
        close = closes[i]
        confidence = confidence_levels[i]

        # --- A) Handle Entry ---
        if up and 0.6 < confidence < 0.8 and position == False:
            # Calculate shares based on available cash
            n_shares = int(cash_held / close)  # Whole shares for realism

            # Only enter if we can buy at least 1 share
            if n_shares > 0:
                total_cost = n_shares * close
                cash_held = cash_held - total_cost  # Cash decreases
                market_value = total_cost * (1 - entry_cost)  # Shares worth less due to fee
                shares_held = n_shares
                position = True

        # --- B) Handle Exit ---
        elif not up and position:
            # Calculate sell fee with min/max
            fee_per_share = 0.000195
            sell_fee = shares_held * fee_per_share
            if sell_fee < 0.01:
                sell_fee = 0.01
            elif sell_fee > 9.79:
                sell_fee = 9.79

            # Realized cash after selling
            gross_proceeds = market_value
            net_proceeds = gross_proceeds * (1 - exit_cost) - sell_fee

            cash_held = cash_held + net_proceeds
            market_value = 0.0
            shares_held = 0
            position = False

        # --- C) Apply Market Return (if invested) ---
        if position:
            # Compounding: value of our shares changes by today's return
            market_value = market_value * (1 + ret)
            # Note: If shares_held > 0, market_value stays > 0.

        # --- D) Record total equity at end of this step ---
        portfolio[i + 1] = cash_held + market_value

    return portfolio


# --- Usage ---
# Ensure your arrays are numpy arrays (not lists)
p_up_np = np.array(p_up, dtype=np.bool_)
confidence_levels_np = np.array(probabilities, dtype=np.float64)
forward_returns_np = np.array(forward_returns, dtype=np.float64)
closes_np = np.array(close_prices, dtype=np.float64)

print(len(p_up_np), len(confidence_levels_np), len(forward_returns_np), len(closes_np))

# Run the optimized backtest
equity_curve = run_backtest(p_up_np, confidence_levels_np, forward_returns_np, closes_np)

period_returns = np.diff(equity_curve) / equity_curve[:-1]  # returns per minute

# --- 2. Sharpe Ratio (Annualized) ---
risk_free_rate_annual = 0.02  # 2% per year

# Number of 1-minute bars in a trading year (approx)
# 252 trading days * 390 minutes (6.5 hours) ≈ 98,280 minutes
periods_per_year = 252 * 390

# Risk-free rate per minute
rf_period = risk_free_rate_annual / periods_per_year

# Excess returns over risk-free
excess_returns = period_returns - rf_period

# Annualized Sharpe (using standard deviation)
if np.std(excess_returns) > 0:
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
else:
    sharpe_ratio = np.nan

print(f'Accuracy: {accuracy}% | {reversal_accuracy}%')

print(f'Recent Accuracy: {recent_accuracy}% | {recent_reversal_accuracy}%')

print(f"Annualized Sharpe Ratio: {sharpe_ratio:.3f}")

# --- 3. Max Drawdown and Other Metrics ---
# Running maximum of the equity curve
running_peak = np.maximum.accumulate(equity_curve)

# Drawdown at each point (percentage from peak)
drawdown_series = (running_peak - equity_curve) / running_peak
max_drawdown = np.max(drawdown_series)

print(f"Max Drawdown: {max_drawdown:.2%}")

# Total Return                     Initial Capital
total_return = (equity_curve[-1] / 1000) - 1
print(f"Total Return: {total_return:.2%}")

avg_win = np.where(period_returns > 0, period_returns, 0)
avg_win = avg_win[avg_win != 0]
avg_win = avg_win.mean()
avg_loss = np.where(period_returns > 0, 0, period_returns)
avg_loss = avg_loss[avg_loss != 0]
avg_loss = avg_loss.mean()

print(f'Average Win:{avg_win:.5f}')
print(f'Average Loss:{avg_loss:.5f}')

n = len(equity_curve) / (252 * 390)

cagr = (equity_curve[-1]/equity_curve[0])**(1/n) - 1

print(f'CAGR:{cagr:.1%}')

# --- 4. Equity Curve Plot ---
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=np.arange(0, len(equity_curve)),
        y=equity_curve,
        mode='lines',
        name='Equity Curve',
        line=dict(color='red', width=2)
    )
)

# Layout
fig.update_layout(
    title='Equity Curve',
    xaxis_title='Time Step',
    yaxis_title='Portfolio Value ($)',
    legend=dict(x=0.02, y=0.98),
    hovermode='x unified'
)

fig.show()
