import sys
from typing import Dict
from cvxopt import matrix
import cvxopt
from cvxopt.solvers import qp
import quantstats as qs
import pandas as pd
import numpy as np
import warnings

cvxopt.solvers.options['show_progress'] = False

def get_proportion_large(mu, cov, gamma=1):
    num = mu.size
    P = gamma * cov
    q = -mu
    G = - np.identity(num)
    h = np.zeros((num, 1))
    A = np.ones((1, num))
    b = np.ones((1, 1))
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = qp(P, q, G, h, A, b)
    x = res['x']
    x = np.array(x).reshape(-1)
    assert abs(x.sum() - 1) < 1e-5, f"{x.sum()} != 1"
    return x

def get_returns_statistic(data: pd.DataFrame):
    """return mean, co-variance"""
    data = data.copy()
    ret : pd.DataFrame = data
    ret = ret.dropna()
    mu = ret.mean()
    cov = ret.cov()
    return mu.to_numpy(), cov.to_numpy()

def quantize(portfolio, total=1000):
    quantized_portfolio = [int(x*total) for x in portfolio]
    quantized_portfolio[2] = total - quantized_portfolio[0] - quantized_portfolio[1]
    quantized_portfolio = [x/total for x in quantized_portfolio]
    return quantized_portfolio

def mean_variance_sliding_portfolio_large(data, gamma, init_mu, init_cov, update_period=1, alpha=1, period_len=None):
    portfolios = [] # strategies
    portfolio = get_proportion_large(init_mu, init_cov, gamma).tolist()
    portfolios.append(portfolio)
    for i in range(1, data.shape[0]):
        if i % update_period == 0:
            # update the porfolio
            start = 0 if period_len is None or i-period_len < 0 else i-period_len
            mu, cov = get_returns_statistic(data.iloc[start:i])
            # update init mu, cov
            init_mu = init_mu*(1-alpha) + alpha*mu if not np.isnan(mu).any() else init_mu
            init_cov = init_cov*(1-alpha) + alpha*cov if not np.isnan(cov).any() else init_cov
            portfolio = get_proportion_large(init_mu, init_cov, gamma).tolist()
        portfolios.append(portfolio)
    portf = pd.DataFrame(portfolios)
    portf.columns = data.columns
    return portf


def mean_variance_sliding_result_large(data, gamma, update_period, alpha=0.5, 
        period_len=None, init_mu=None, init_cov=None):
    """
    Construct a portfolio based on mean-variance optimization and return the result on data.

    Args:
        * ```data```: a DataFrame which contains all available strategies' net value.
        * ```portfolio_info```: Dict[pd.DataFrame], a dictionary of DataFrames. The keys of
            the dictionary are the names of the strategies, which are also the keys in ```data```.
            Each DataFrame contains proportion of each component at each day.
        * ```gamma```: tolerance of risk. Larger ```gamma``` means more risk adversed.
        * ```update_period```: number of days before reweighting the portfolio.
        * ```alpha```: a internal parameter for updating the estimated mean and covariance (between 0 and 1).
        * ```period_len```: length of the period that is used to estimate the mean and covariance.
        * ```plot```: whether to generate a report using QuantStat.
        * ```init_mu```: np.ndarray, an initial guess of the expected daily return for each strategy.
        * ```init_cov```: np.ndarray, an initial guess of the covariance of the daily return for each strategy.
        * ```return_commission```: whether to return the commission fee also.
    
    Return: a DataFrame containing the net value of the portfolio.

    Reminder: ```data``` should have a column of all zero value with the column name 'cash'.
    """
    num = data.shape[1] # number of available strategies
    if init_mu is None:
        init_mu = np.random.normal(0.001, 0.001, num)
    if init_cov is None:
        init_cov = np.diag(np.random.normal(1e-4, 1e-5, num)) + np.random.normal(5e-5, 1e-5, (num, num))
        init_cov = np.abs(init_cov)
    data = data.copy()
    portfolio = mean_variance_sliding_portfolio_large(data, gamma, init_mu, init_cov, update_period, alpha, period_len)
    portfolio.index = data.index
    return portfolio


if __name__ == '__main__':

    portfolio_info = {
        'bitcoin': pd.DataFrame([[1, 0, 0]]*10, columns=['bitcoin', 'gold', 'cash']),
        'gold': pd.DataFrame([[0, 1, 0]]*10, columns=['bitcoin', 'gold', 'cash']),
        'cash': pd.DataFrame([[0, 0, 1]]*10, columns=['bitcoin', 'gold', 'cash']),
    }
    print(portfolio_info)
    sys.exit()
    # run portfolio
    result, portfolio = mean_variance_sliding_result_large(data, portfolio_info, 1, 20, 0.6, 50, True, 
                                                           return_portfolio_weight=True,
                                                           benchmark='bitcoin')
    print(result)
    print(portfolio)