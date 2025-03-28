import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from yfinance import download  # Ensure yfinance is installed: pip install yfinance

# Step 1: Fetch Real S&P 500 Data
data = download('^GSPC', start='2000-01-01', end='2025-03-27')
print("Data columns:", data.columns)  # Debug: Check available columns
if 'Close' not in data.columns:
    raise KeyError("Close column not found in data!")
prices = data['Close']  # Use closing prices
returns = prices.pct_change().dropna() * 100  # Daily returns in percentage
prices = prices.iloc[1:]  # Align prices with returns

# Step 2: Fit Markov-Switching Model
model = sm.tsa.MarkovRegression(
    returns,
    k_regimes=2,
    switching_variance=True
)
results = model.fit()

# Step 3: Extract Results
print("Transition Matrix:")
print(results.params[['p[0->0]', 'p[1->0]']])
print("\nRegime Parameters:")
print(results.params[['const[0]', 'const[1]', 'sigma2[0]', 'sigma2[1]']])

# Step 4: Smoothed Probabilities and State Changes
smoothed_probs = results.smoothed_marginal_probabilities
bull_prob = smoothed_probs[0]  # Regime 0 (assumed bull)
bear_prob = smoothed_probs[1]  # Regime 1 (assumed bear)

# Determine dominant regime (0 = bull, 1 = bear)
regime = (bear_prob > bull_prob).astype(int)  # Length: 6344
# Detect state changes (where regime shifts from 0 to 1 or 1 to 0)
regime_diff = regime.diff()  # Length: 6344, first entry is NaN
state_changes = (regime_diff != 0) & (regime_diff.notna())  # True at switches, False at NaN
change_dates = regime.index[state_changes]  # Dates where regime switches

# Step 5: Plot Results with Price Chart and State Changes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Top subplot: Bull and Bear Probabilities
ax1.plot(returns.index, bull_prob, label='Bull Regime Probability', color='green')
ax1.plot(returns.index, bear_prob, label='Bear Regime Probability', color='red')
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
ax1.set_title('Markov Regime-Switching: S&P 500 Bull and Bear Probabilities (2000-2025)')
ax1.set_ylim(0, 1)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Bottom subplot: S&P 500 Price with State Changes
ax2.plot(prices.index, prices, label='S&P 500 Close', color='blue')
for change_date in change_dates:
    ax2.axvline(x=change_date, color='black', linestyle='--', alpha=0.5, label='Regime Switch' if change_date == change_dates[0] else None)
ax2.set_title('S&P 500 Price with Regime Switches')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Step 6: Estimate Probability of Reaching a Top
latest_bull_prob = bull_prob.iloc[-1]
latest_bear_prob = bear_prob.iloc[-1]
print(f"Latest Bull Probability: {latest_bull_prob:.3f}")
print(f"Latest Bear Transition Probability (P12): {results.params['p[0->1]']:.3f}")
if latest_bull_prob < 0.5 and results.params['p[0->1]'] > 0.1:
    print("Warning: Potential market top detected!")