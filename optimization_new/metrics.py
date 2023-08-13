import numpy as np
#from .. import ced
from scipy import stats

# all measures are vectorized to work with Problem (rather than Elementwise Problem)
# returns are assumed to be of size j x n, where j is the time and n is the population size 
# weights are assumed to be of size n x k, where k is the number of assets
# result of evaluate methods should be n x 1


# returns have already been weighted by the weights
# Risk Measures

class Metric:
     def post_process(self, results):
         return results


#class CED(Metric):
#    def __init__(self, window = 12, alpha = 0.95):
#        self.uses_returns = True
#        
#        self.window = window
#        self.alpha = alpha
#        self.name = "{}M {}% CED".format(window, int(100 * alpha))
        
        
#    def evaluate(self, returns, weights):
#        return np.apply_along_axis(ced.calculate_ced_fast, 0, returns, 
#                                   window = self.window, alpha = self.alpha, is_arr = True)
    
    
class Volatility(Metric):
    
    def __init__(self, cov_matrix, annualization = "Sqrt", periodicity = 12):
        
        self.uses_returns = False
        self.cov_matrix = cov_matrix
        self.name = "Volatility"
        self.periodicity = periodicity
        
        if annualization == "Sqrt":
            self.ann_func = annualize_var
        else:
            self.ann_func = lambda x, _: np.sqrt(x)
            
        
    def evaluate(self, returns, weights):
        return np.einsum('ij,ij->i', weights @ self.cov_matrix, weights)
    
    
    def post_process(self, results):
        # annualized volatility
        return self.ann_func(results, periodicity = self.periodicity)
    

class Historical_CVaR(Metric):
    def __init__(self, alpha = 0.95):
        self.uses_returns = True
        self.alpha = alpha
        self.name = "{}% CVaR".format(int(alpha * 100))
    
    def evaluate(self, returns, weights):
        var = np.quantile(returns, self.alpha, axis = 0)
        return returns[returns < var].mean(axis = 0)
    
    
class Historical_VaR(Metric):
    def __init__(self, alpha = 0.95):
        self.uses_returns = True
        self.alpha = alpha
        self.name = "{}% VaR".format(int(alpha * 100))
    
    def evaluate(self, returns, weights):
        return np.quantile(returns, self.alpha, axis = 0)
    
class DownsideDeviation(Metric):
    def __init__(self, periodicity = 12):
        self.uses_returns = True
        self.name = 'Downside Deviation'
        self.periodicity = periodicity
 
        
    def evaluate(self, returns, weights):
        downside_only = np.where(returns < 0, 0, returns)
        return np.mean(downside_only ** 2, axis = 0)
    
    def post_process(self, results):
        return annualize_var(results, periodicity = self.periodicity)

        

    
    
# Return Measures
    
    
    
class AvgGeoReturn(Metric):
    def __init__(self, annualization = "Compound", periodicity = 12):
        self.uses_returns = True
       
    
        if annualization == "Compound":
            self.ann_return_func = annualize_return_compound
        elif annualization == "Simple":
            self.ann_return_func = annualize_return_simple
        else:
            self.ann_return_func = no_annualize # no annualization
            
            
        self.name = "Avg Geo Return"
        self.periodicity = 12
        
    def evaluate(self, returns, weights):
        # returns is now assumed to be a j X n array, where n is the population size and j is the history size
        
        cumulative_returns = (1 + returns).cumprod(axis = 0)[-1]
        return -1 * cumulative_returns ** (1 / returns.shape[0])
    
    def post_process(self, results):
        return self.ann_return_func(-1 * results, periodicity=self.periodicity)
    

    
class AvgArithReturn(Metric):
    
    def __init__(self, mu, annualization = "Compound", periodicity = 12):
        self.mu = mu
        self.uses_returns = False
        
        if annualization == "Compound":
            self.ann_return_func = annualize_return_compound
        elif annualization == "Simple":
            self.ann_return_func = annualize_return_simple
        else:
            self.ann_return_func = no_annualize # no annualization
        
        
        self.name = "Avg Arithmetic Return"
        self.periodicity = periodicity
        
    def evaluate(self, returns, weights):
    
        # weights is n X k numpy array
        return -1 * self.mu @ weights.T
        
    def post_process(self, results):
        return self.ann_return_func(-1 * results + 1, 
                                    periodicity = self.periodicity)

    

    
    
    
# Regularization and miscellaneous

    
class L2_Reg(Metric):
    def __init__(self):
        self.uses_returns = False
        self.name = "L2 Reg"
        
    def evaluate(self, returns, weights):
        return np.sum(weights ** 2, axis = 1)
    
    
class Risk_Equality(Metric):
    def __init__(self, cov_matrix, handle_neg_contr = "ignore"):
        self.uses_returns = False
        self.cov_matrix = cov_matrix
        self.name = "Equality of Volatility Contr"
        self.handle_neg_contr = handle_neg_contr
    
    def evaluate(self, returns, weights):
        
        # Compute portfolio variances
        portfolio_variances = np.einsum('ij,ij->i', weights @ self.cov_matrix, weights)

        # Compute the marginal contribution to risk of each portfolio
        MCR = (weights @ self.cov_matrix) / np.sqrt(portfolio_variances)[:, None]

        # Compute the contribution to volatility of each portfolio
        contributions = weights * MCR

        # minimizing variance of risk contributions
        
        if self.handle_neg_contr == "ignore":
            contributions[contributions < 0] = np.nan
            
        # other option is to include, which could include negative vol contributions in variance
        
        
        return np.nanvar(contributions, axis=1)
    
    
class Liquidity(Metric):
    def __init__(self, scores):
        self.uses_returns = False
        self.scores = scores
        self.name = "Liquidity"
    
    def evaluate(self, returns, weights):
        return -1 * weights @ self.scores.T
    
    def post_process(self, results):
        return -1 * results
    
    
    
class Skewness(Metric):
    def __init__(self):
        self.uses_returns = True
        self.name = "Skew"
    
    def evaluate(self, returns, weights):
        # maximizing skew
        return -1 * stats.skew(returns, axis = 0)
    
    def post_process(self, results):
        return -1 * results
    

# merge bull and bear beta into single metric Beta, which takes condition as keyword argument
class BearBeta(Metric):
    def __init__(self, returns, risk_index, thresh = 0, units = "Return", bear_betas = None):
        self.uses_returns = False
        self.name = "Bear Beta"
        
        if bear_betas is not None:
            self.bear_betas = bear_betas.values
        else:
        
            if units == "Std":
                thresh = risk_index.mean() - (thresh * risk_index.std())
        
        
            down_months = risk_index < thresh
            down_returns = returns[down_months]
            down_index = risk_index[down_months]
            
            self.bear_betas = down_returns.apply(beta, index = down_index, axis = 0).values
        
        
    def evaluate(self, returns, weights):
        
        # linear with respect to the weights
        return weights @ self.bear_betas.T
    
    
class BullBeta(Metric):
    def __init__(self, returns, risk_index, bull_betas = None):
        self.uses_returns = False
        self.name = "Bull Beta"
        
        if bull_betas is not None:
            self.bull_betas = bull_betas.values
        else:
            
            up_months = risk_index > 0
            up_months = returns[up_months]
            up_index = risk_index[up_months]

            self.bull_betas = up_months.apply(beta, index = up_index, axis = 0).values
        
        
    def evaluate(self, returns, weights):
        
        # linear with respect to the weights
        return -1 * weights @ self.bear_betas.T
    
    def post_process(self, results):
        return -1 * results
    
    
    
class Prob_Neg_Month(Metric):
    def __init__(self):
        self.uses_returns = True
        self.name = "Prob of Neg Month"
    def evaluate(self, returns, weights):
        return np.mean(returns < 0, axis = 0)
    
    
# efficienty statistics

class SharpeRatio(Metric):
    
    def __init__(self, avg_return_metric, volatility_metric, periodicity = 12, rf_rate = 0.02):
        
        # expects already initialized return and volatility sub metrics
        # return can be geometric or arithmetic
        
        self.uses_returns = (avg_return_metric.uses_returns) or (volatility_metric.uses_returns)
        self.name = "Sharpe Ratio"
        self.rf_rate = rf_rate
        
        self.mean_submetric = avg_return_metric
        self.volatility_submetric = volatility_metric
            
            
    def evaluate(self, returns, weights):
        
        
        # we have to annualize return and volatility to get a meaningful sharpe ratio
        numerator = self.mean_submetric.post_process(self.mean_submetric.evaluate(returns, weights)) - self.rf_rate
        denominator = self.volatility_submetic.post_process(self.volatility_submetric.evaluate(returns, weights))
        
        return -1 * numerator / denominator

    def post_process(self, results):
        return -1 * results
    

class SortinoRatio(Metric):
    
    def __init__(self, avg_return_metric, downside_deviation_metric, periodicity = 12, target_rate = 0.02):\
        
        self.uses_returns = (avg_return_metric.uses_returns) or (downside_deviation_metric.uses_returns)
        self.name = "Sortino Ratio"
        self.target_rate = target_rate
        
        self.mean_submetric = avg_return_metric
        self.downside_deviation_submetric = downside_deviation_metric
    
    def evaluate(self, returns, weights):
        
        numerator = self.mean_submetric.post_process(self.mean_submetric.evaluate(returns, weights)) - self.rf_rate
        denominator = self.downside_deviation_submetric.post_process(self.downside_deviation_submetric.evaluate(returns, weights))
        
        return -1 * numerator / denominator
    
    def post_process(self, results):
        return -1 * results
    
        

            
        


# annualization function

def beta(asset, index):
    return np.cov(asset, index)[0, 1] / np.var(index)

def annualize_var(var, periodicity = 12):
    # sqrts and annualizes variance
    # same as np.sqrt(var) * np.sqrt(periodicity)
    
    return np.sqrt(var * periodicity)
    
        
def annualize_sharpe(sr, periodicity = 12):
    # getting from monthly SR to annualized
    # has to assume simple multiplicative model
    
    return sr * (periodicity / np.sqrt(periodicity))
    
    
def no_annualize(r, periodicity = 12):
    return r - 1

def annualize_return_compound(r, periodicity = 12):
    # expects r to be in the 1 + r format
    return (r ** periodicity) - 1

def annualize_return_simple(r, periodicity = 12):
    # expects r to be in the 1 + r format
    return (r - 1) * periodicity
    
    
        
        