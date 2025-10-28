import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, binomtest
import datetime as dt

"""
 Portfolio Value at Risk (VaR) Simulation and Validation

This script implements a risk measurement system (VaR) for
a portfolio of equity securities, using two methodologies:
1. Historical VaR
2. Monte Carlo VaR

It also includes a backtesting phase (Kupiec's POF Test) to validate
the accuracy of the historical VaR model.
"""

def fetch_data(tickers, start_date, end_date):
    """
    Downloads adjusted close prices for the specified tickers.
    """
    try:
        # Set auto_adjust=False to ensure 'Adj Close' column is present
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        
        # Handle missing data (VERY IMPORTANT for illiquid/OTC tickers)
        # 1. Forward-fill: fills NaNs with the last valid price
        #    (assumes the price didn't change if there was no trade)
        # data.fillna(method='ffill', inplace=True) # Old syntax
        data.ffill(inplace=True) # New syntax
        
        # 2. Back-fill: fills any remaining NaNs at the beginning of the dataset
        # data.fillna(method='bfill', inplace=True) # Old syntax
        data.bfill(inplace=True) # New syntax
        
        # We only remove rows where *all* tickers are NaN (e.g., full holidays)
        data.dropna(how='all', inplace=True)

        if data.empty:
            print(f"Error: No data downloaded or insufficient data (after dropna/fillna) for tickers {tickers}. Check symbols or date range.")
            return None
        return data
    except Exception as e:
        print(f"Error during data download: {e}")
        return None

def calculate_historical_var(portfolio_returns, alpha=0.01):
    """
    Calculates the historical Value at Risk (VaR).

    Args:
        portfolio_returns (pd.Series): Series of historical portfolio returns.
        alpha (float): Confidence level (e.g., 0.01 for 99% VaR).

    Returns:
        float: VaR value (as a positive number).
    """
    # VaR is the 'alpha' percentile of the returns distribution.
    # We use np.percentile, which takes a value between 0 and 100.
    # VaR is a loss, so we take the low percentile (e.g., 1st)
    # and return it as a positive number.
    var_value = -np.percentile(portfolio_returns, 100 * alpha)
    return var_value

def calculate_monte_carlo_var(log_returns, weights, alpha=0.01, T=1, n_sims=10000):
    """
    Calculates Value at Risk (VaR) using Monte Carlo simulation.

    Args:
        log_returns (pd.DataFrame): DataFrame of historical log returns of assets.
        weights (np.array): Array of portfolio weights.
        alpha (float): Confidence level.
        T (int): Time horizon (in days).
        n_sims (int): Number of simulations.

    Returns:
        tuple: (Monte Carlo VaR, array of simulated returns)
    """

    # 1. Calculate historical distribution parameters
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()

    # 2. Cholesky decomposition to correlate random shocks
    # L * L.T = cov_matrix
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        print("Error: Covariance matrix is not positive definite. Cannot perform Cholesky decomposition.")
        # Fallback: use a diagonal matrix (ignores correlations)
        cov_matrix_diag = np.diag(np.diag(cov_matrix))
        L = np.linalg.cholesky(cov_matrix_diag)

    # 3. Run the simulation

    # Array to store simulation results
    simulated_portfolio_returns = np.zeros(n_sims)

    for i in range(n_sims):
        # Generate uncorrelated random shocks (Z)
        # Z will have shape (n_assets, T)
        Z = np.random.normal(size=(len(weights), T))

        # Correlate shocks using Cholesky (L @ Z) and apply mean
        # daily_sim_returns will have shape (n_assets, T)
        daily_sim_returns = mean_returns.values.reshape(-1, 1) + L @ Z

        # Calculate simulated portfolio return for this iteration
        # (Weighted sum of asset returns over the time horizon T)
        # weights is (n_assets,), daily_sim_returns is (n_assets, T)
        # Result of np.dot(weights, daily_sim_returns) is (T,)
        # Sum over T to get the total return for the simulation
        sim_portfolio_return = np.dot(weights, daily_sim_returns).sum()

        simulated_portfolio_returns[i] = sim_portfolio_return

    # 4. Calculate VaR from the simulated distribution
    var_mc = -np.percentile(simulated_portfolio_returns, 100 * alpha)

    return var_mc, simulated_portfolio_returns

def backtest_var(portfolio_returns, var_estimate, alpha):
    """
    Performs VaR backtesting using Kupiec's "Proportion of Failures" (POF) test.

    Args:
        portfolio_returns (pd.Series): Actual historical returns.
        var_estimate (float): VaR estimate (positive number).
        alpha (float): Confidence level used for VaR.

    Returns:
        tuple: (Number of exceptions, Failure rate, p-value of Kupiec's test)
    """
    # An exception occurs when the actual loss exceeds the VaR estimate
    # (i.e., the return is less than -var_estimate)
    exceptions = portfolio_returns[portfolio_returns < -var_estimate]
    n_exceptions = len(exceptions)
    n_obs = len(portfolio_returns)

    if n_obs == 0:
        return 0, 0, 1.0 # Avoid division by zero

    failure_rate = n_exceptions / n_obs

    # Kupiec's POF test (binomial test)
    # H0: The observed failure rate is consistent with alpha.
    # H1: The observed failure rate is different from alpha.

    # We use binomtest to calculate the p-value
    # If p-value < 0.05, we reject H0 (the model is inaccurate)
    # If p-value > 0.05, we do not reject H0 (the model is acceptable)
    if n_exceptions == 0 and n_obs > 0:
        # If there are no exceptions, the p-value is (1-alpha)^n_obs
        # binomtest handles this case
        pass

    try:
        test_result = binomtest(n_exceptions, n_obs, p=alpha, alternative='two-sided')
        p_value = test_result.pvalue
    except ValueError as e:
        print(f"Error in backtesting: {e}")
        p_value = 1.0 # Not conclusive

    return n_exceptions, failure_rate, p_value

def plot_results(hist_returns, mc_returns, var_hist, var_mc, alpha):
    """
    Displays the results: distributions and VaR values.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Confidence title
    conf_level_str = f"{100*(1-alpha):.0f}%"

    # Plot 1: Historical VaR
    ax1.hist(hist_returns, bins=50, alpha=0.7, label='Historical Returns', color='navy', density=True)
    ax1.axvline(x=-var_hist, color='red', linestyle='--', linewidth=2,
                label=f'Historical VaR {conf_level_str}: {-var_hist:.4f}')
    ax1.set_title('Historical Distribution and VaR', fontsize=16)
    ax1.set_xlabel('Portfolio Returns')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot 2: Monte Carlo VaR
    ax2.hist(mc_returns, bins=50, alpha=0.7, label='Simulated Returns (MC)', color='teal', density=True)
    ax2.axvline(x=-var_mc, color='orange', linestyle='--', linewidth=2,
                label=f'Monte Carlo VaR {conf_level_str}: {-var_mc:.4f}')
    ax2.set_title('Monte Carlo Distribution and VaR', fontsize=16)
    ax2.set_xlabel('Simulated Portfolio Returns')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('Portfolio Value at Risk (VaR) Analysis', fontsize=20, y=1.02)
    plt.tight_layout()

    # Save the chart
    plt.savefig('var_simulation_results.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'var_simulation_results.png'")
    plt.show()

def main():
    """
    Main function to run the VaR analysis.
    """
    print("Starting Portfolio VaR simulation and validation...")

    # --- Portfolio Configuration ---
    # Custom portfolio as per request
    tickers = [
        'INVMF', 'GOOGL', 'PWQQF', 'NVDA', 'ORCL', 'BFOR', 'XTMWF', 
        'AAPL', 'IJH', 'TSLA', 'NTDOF', 'GLD', 'MSFT', 'GS'
    ]
    # Weights as a percentage of the total portfolio (sum 86%, 14% is cash)
    weights_list = [
        0.21, 0.11, 0.10, 0.10, 0.09, 0.08, 0.05, 
        0.04, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01
    ]
    weights = np.array(weights_list)
    # n_assets = len(tickers) # No longer needed for weights

    # --- Simulation Parameters ---
    start_date = '2020-01-01'
    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    alpha = 0.01  # 99% confidence level (alpha = 1 - 0.99)
    T = 1 # Time horizon (in days) for VaR calculation
    n_sims = 10000 # Number of Monte Carlo simulations
    initial_investment = 1_000_000 # Hypothetical investment in EUR/USD

    # 1. Get Data
    print(f"\n1. Downloading data for {tickers} from {start_date} to {end_date}...")
    prices = fetch_data(tickers, start_date, end_date)
    if prices is None:
        print("Aborting analysis due to data issues.")
        return

    # 2. Calculate Logarithmic Returns
    # Logarithmic returns are preferred for stochastic analysis
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    if log_returns.empty:
        print("Error: No return data available after processing. Check source data.")
        return

    # 3. Calculate Historical Portfolio Returns
    # (Portfolio return = weighted sum of asset returns)
    # This calculates the total portfolio return, assuming
    # the 14% cash has a return of 0.
    portfolio_log_returns = log_returns.dot(weights)

    print("\n--- Value at Risk (VaR) Results ---")
    print(f"Portfolio: {tickers}")
    print(f"Weights: {weights}")
    print(f"Period: {start_date} - {end_date} ({len(log_returns)} observations)")
    print(f"Hypothetical Investment: {initial_investment:,.0f}")
    print(f"Confidence Level: {100*(1-alpha):.0f}%")
    print(f"Time Horizon (T): {T} day(s)")
    print("-" * 40)

    # 4. Calculate Historical VaR
    # Historical VaR is typically for a single period (e.g., 1 day)
    var_historical = calculate_historical_var(portfolio_log_returns, alpha)
    var_amount_hist = var_historical * initial_investment
    print(f"Historical VaR (1-day): {var_historical:.4f} (or {var_amount_hist:,.2f} EUR/USD)")

    # 5. Calculate Monte Carlo VaR
    # Monte Carlo VaR can be for T days, but the simulation needs to reflect that
    # For T=1 day, the simulation should give daily returns
    var_mc, sim_returns = calculate_monte_carlo_var(log_returns, weights, alpha, T=T, n_sims=n_sims)
    var_amount_mc = var_mc * initial_investment
    print(f"Monte Carlo VaR ({n_sims} sims, T={T} days): {var_mc:.4f} (or {var_amount_mc:,.2f} EUR/USD)")
    print("-" * 40)

    # 6. Backtesting (on Historical VaR)
    print("\n--- Model Backtesting (Historical VaR) ---")
    # Backtesting is typically done on 1-day ahead predictions
    n_exc, rate, p_val = backtest_var(portfolio_log_returns, var_historical, alpha)

    print(f"Confidence Level (Alpha): {alpha*100:.2f}%")
    print(f"Total Observations (N): {len(portfolio_log_returns)}")
    print(f"Observed Exceptions (x): {n_exc}")
    print(f"Observed Failure Rate: {rate*100:.2f}%")

    print(f"\nKupiec's POF Test (p-value): {p_val:.4f}")
    if p_val < 0.05:
        print("Result: REJECTED (p-value < 0.05). The model is inaccurate.")
    else:
        print("Result: ACCEPTED (p-value >= 0.05). The model is considered accurate.")
    print("-" * 40)

    # 7. Visualization
    print("\nGenerating charts...")
    # Plotting simulated returns for T days might require adjustment if T > 1
    # Currently plotting total return over T days for MC
    plot_results(portfolio_log_returns, sim_returns, var_historical, var_mc, alpha)
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
