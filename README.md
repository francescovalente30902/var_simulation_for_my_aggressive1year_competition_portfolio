# var_simulation_for_my_aggressive1year_competition_portfolio
A Python script to simulate and validate portfolio Value at Risk (VaR) using Historical and Monte Carlo methods, including Kupiec's POF backtest for validation.

This Python script performs a simulation and validation of Value at Risk (VaR) for a portfolio of financial assets. It uses two different methodologies for calculating VaR and includes a statistical backtest to validate the model's accuracy.

Main Features

Automatic Data Retrieval: Downloads historical price data from Yahoo Finance (yfinance).

Dirty Data Handling: Manages illiquid tickers (e.g., OTC) and non-trading days by using a forward-fill (ffill) strategy to prevent data loss.

Dual VaR Methodology:

Historical VaR: Calculates VaR based on the percentile of the historical returns distribution.

Monte Carlo VaR: Simulates thousands of possible future returns (using Cholesky decomposition) to estimate VaR.

Statistical Backtesting: Performs the Kupiec's "Proportion of Failures" (POF) test to statistically validate whether the number of exceptions (days when the actual loss exceeded the VaR) is consistent with the chosen confidence level.

Visualization: Generates and saves a chart (var_simulation_results.png) showing the historical and simulated return distributions, highlighting the VaR thresholds.

Methodologies

1. Historical VaR

This non-parametric approach makes no assumptions about the distribution of returns. It simply sorts the historical portfolio returns (from worst to best) and identifies the return corresponding to the alpha percentile (e.g., 1% for a 99% VaR).

2. Monte Carlo VaR

This parametric approach assumes that returns follow a distribution (in this case, a multivariate normal).

Calculates the mean and covariance matrix of the historical log returns of the assets.

Uses Cholesky decomposition to generate correlated random shocks.

Simulates thousands (n_sims) of possible daily portfolio returns.

Calculates VaR as the percentile of this simulated distribution.

3. Backtesting (Kupiec's POF Test)

This is a crucial test to validate the VaR model's accuracy. It compares the expected number of failures (based on alpha) with the observed number of failures (days when the actual loss exceeded the VaR).

Null Hypothesis (H0): The model is accurate (the observed failure rate is statistically equal to alpha).

Result (p-value):

p-value >= 0.05: ACCEPTED. There is no statistical evidence to reject the model. It is considered accurate.

p-value < 0.05: REJECTED. The model is inaccurate (it systematically underestimates or overestimates risk).

Requirements

Ensure you have the following Python libraries installed:

pip install pandas numpy yfinance matplotlib scipy


How to Run

Save the code as var_simulation_updated.py.

Install the requirements listed above.

Run the script from your command line:

python var_simulation_updated.py


How to Customize the Portfolio

To analyze your own portfolio, modify the tickers and weights_list variables inside the main() function:

# --- Portfolio Configuration ---
# Replace with your tickers
tickers = [
    'MY_TICKER_1', 'MY_TTICKER_2', 'MY_TICKER_3', ...
]
# Replace with the corresponding weights (in decimal format)
# The sum does not necessarily have to be 1 (the missing part is considered cash)
weights_list = [
    0.25, 0.50, 0.25, ...
]

# ...you can also change the alpha (e.g., 0.05 for 95%) or the investment
alpha = 0.01 
initial_investment = 1_000_000 


Output

Running the script will produce two outputs:

Console Output: A detailed summary that includes portfolio parameters, VaR values (in both % and monetary value), and the complete results of the Kupiec backtest.

Image File: A file named var_simulation_results.png will be saved in the same directory. This chart shows the return distributions and VaR thresholds.
