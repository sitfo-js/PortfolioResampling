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
#        self.is_pct = True


#    def evaluate(self, returns, weights):
#        return np.apply_along_axis(ced.calculate_ced_fast, 0, returns,
#                                   window = self.window, alpha = self.alpha, is_arr = True)



class Annualizable_Metric(Metric):
    def __init__(self, kind, periodicity = 12):
        self.periodicity = periodicity

        if kind == "None":
            self.annualize_func = self.no_annualize

        if kind == "Simple":
            self.annualize_func = self.simple

        if kind == "Compound":
            self.annualize_func = self.compound

        if kind == "Sqrt":
            self.annualize_func = self.annualize_std


    def no_annualize(self, r):
        return r - 1

    def simple(self, r):
        # simple multiplicate annualization
        return (r - 1) * self.periodicity


    def compound(self, r):
        # expects r to be in the 1 + r format
        return (r ** self.periodicity) - 1


    def annualize_std(self, s):
        # annualizes standard deviation

        return s * np.sqrt(self.periodicity)

    def post_process(self, results):
        return self.annualize_func(results)

    
    
class Volatility(Annualizable_Metric):
    
    def __init__(self, cov_matrix, annualization = "Sqrt", periodicity = 12):
        
        self.uses_returns = False
        self.is_pct = True
        self.cov_matrix = cov_matrix
        self.name = "Volatility"

        super().__init__(kind = "Sqrt", periodicity = periodicity)
        
    def evaluate(self, returns, weights):
        return np.sqrt(np.einsum('ij,ij->i', weights @ self.cov_matrix, weights))
    

class Historical_CVaR(Metric):
    def __init__(self, alpha = 0.95):
        self.uses_returns = True
        self.is_pct = True
        self.alpha = alpha
        self.name = "{}% CVaR".format(int(alpha * 100))
    
    def evaluate(self, returns, weights):
        var = np.quantile(returns, self.alpha, axis = 0)
        return returns[returns < var].mean(axis = 0)
    
    
class Historical_VaR(Metric):
    def __init__(self, alpha = 0.95):
        self.uses_returns = True
        self.is_pct = True
        self.alpha = alpha
        self.name = "{}% VaR".format(int(alpha * 100))
    
    def evaluate(self, returns, weights):
        return np.quantile(returns, self.alpha, axis = 0)
    
class DownsideDeviation(Annualizable_Metric):
    def __init__(self, periodicity = 12):
        self.uses_returns = True
        self.is_pct = True
        self.name = 'Downside Deviation'

        super().__init__(kind = "Sqrt", periodicity = periodicity)
 
        
    def evaluate(self, returns, weights):
        # 0s are included in the standard deviation
        return np.where(returns < 0, 0, returns).std(axis = 0)

        

    
    
# Return Measures
    
class AvgGeoReturn(Metric):
    def __init__(self, ann_kind = "Compound", periodicity = 12):
        self.uses_returns = True
        self.is_pct = True
        self.name = "Avg Geo Return"

        super().__init__(kind = ann_kind, periodicity = periodicity)
        
    def evaluate(self, returns, weights):
        # returns are j x n (j = time, n = pop)
        cumulative_returns = (1 + returns).cumprod(axis = 0)[-1]
        return -1 * cumulative_returns ** (1 / returns.shape[0])
    
    def post_process(self, results):
        return super().post_process(-1 * results)
    

    
class AvgArithReturn(Metric):
    
    def __init__(self, mu, ann_kind = "Compound", periodicity = 12):
        self.mu = mu
        self.uses_returns = False
        self.is_pct = True
        self.name = "Avg Arithmetic Return"

        super().__init__(kind = ann_kind, periodicity = periodicity)
        
    def evaluate(self, returns, weights):


        # (1 x k) @ (n x k).T
        # negative to minimize
        return -1 * self.mu @ weights.T
        
    def post_process(self, results):
        return super().post_process(-1 * results + 1)

    

# Regularization and miscellaneous

    
class L2_Reg(Metric):
    # encourages equal weighting (this minimizes sum of squared weights)
    def __init__(self):
        self.uses_returns = False
        self.is_pct = False
        self.name = "L2 Reg"
        
    def evaluate(self, returns, weights):
        return np.sum(weights ** 2, axis = 1)
    
    
class Risk_Equality(Metric):
    def __init__(self, cov_matrix, handle_neg_contr = "ignore"):
        self.uses_returns = False
        self.is_pct = False
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
        
        
        return np.nanstd(contributions, axis=1)
    
    
class Liquidity(Metric):
    def __init__(self, scores):
        self.uses_returns = False
        self.is_pct = True
        self.scores = scores
        self.name = "Liquidity"
    
    def evaluate(self, returns, weights):
        return -1 * weights @ self.scores.T
    
    def post_process(self, results):
        return -1 * results
    
    
    
class Skewness(Metric):
    def __init__(self):
        self.uses_returns = True
        self.is_pct = False
        self.name = "Skew"
    
    def evaluate(self, returns, weights):
        # maximizing skew
        return -1 * stats.skew(returns, axis = 0)
    
    def post_process(self, results):
        return -1 * results
    

class Beta(Metric):

    # add check that returns and risk index are the same size
    # betas should be pd.Series
    # direction indicates whether you want to maximize or minimize beta, default is minimize
    def __init__(self, betas = None, returns = None, risk_index = None, condition_func = None, direction = -1):

        self.uses_returns = False
        self.is_pct = False
        self.name = "Beta" if condition_func is None else "Conditional Beta"
        self.direction = direction


        if betas is not None:
            self.betas = betas
        else:
            assert (returns is None) or (risk_index is None), "Should Provide either Beta assumptions or returns and a risk index"

            if condition_func is not None:

                filter = condition_func(risk_index)
                returns = returns[filter]
                risk_index = risk_index[filter]

            self.betas = returns.apply(self.beta_func, risk_index = risk_index, axis = 0).values



    def beta_func(self, return_series, risk_index):
        return np.cov(return_series, risk_index)[0, 1] / np.var(risk_index)

    def evaluate(self, returns, weights):

        # linear with respect to the weights
        # (n x k) @ (k x 1)
        return weights @ self.betas.T

    def post_process(self, results):
        # flipping twice, because flipping once is for default maximization
        return -1 * self.direction * results



class TrackingError(Annualizable_Metric):
    def __init__(self, risk_index, periodicity = 12):
        self.uses_returns = True
        self.is_pct = True
        self.name = "Tracking Error"
        self.risk_index = risk_index

        super().__init__(kind = "Sqrt", periodicity = periodicity)

    def evaluate(self, returns, weights):
        excess = returns - self.risk_index
        return excess.std(axis = 0)






    
class ProbLessThan(Metric):
    def __init__(self, thresh = 0):
        self.uses_returns = True
        self.is_pct = True
        self.name = "Prob of Month < {}".format(thresh)
        self.thresh = thresh

    def evaluate(self, returns, weights):
        return np.mean(returns < self.thresh, axis = 0)
    
    
