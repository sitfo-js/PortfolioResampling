from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.dnsga2 import DNSGA2
import numpy as np
from . import constraints
from . import utils


# for constraints, still need to ensure that constrained column is in the first position

# make code handle np array inputs or dataframe inputs

# define __repr__ so we are not subject to pymoo's repr


class BasePortfolioProblem(Problem):
    # base class for optimization problems
    
    def __init__(self, n_var, n_obj, metrics, constraint_obj = constraints.LongOnly(), **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0, **kwargs)

        # metrics should be a list of already initialized metric objects
        self.constraint_obj = constraint_obj
        self.should_weight_returns = any([m.uses_returns for m in metrics])
        self.metrics = metrics
        self.metric_names = [m.name for m in metrics]
        
    def _evaluate(self, weights, out, *args, **kwargs):
        out["F"] = self._evaluate_return(weights)

    def post_process(self, weights, F):
        weights = self.constraint_obj.enforce(weights)
        new_F = np.empty_like(F)
        
        for i in range(F.shape[1]):
            new_F[:, i] = self.metrics[i].post_process(F[:, i])
            
        return weights, new_F
    
    def optimize(self, n_pop = 100, n_gen = 100, seed = None, algorithm = DNSGA2):
        
        # initializes algorithm
        algo_instance = algorithm(pop_size = n_pop)
        
        # results
        res = minimize(self, algo_instance, ("n_gen", n_gen), seed = seed, verbose=False)
        
        # extracts and evaluated metrics and post processes
        weights, F = self.post_process(*res.opt.get("X", "F"))
        self.weights = utils.pad_array(weights, (n_pop, weights.shape[1]))
        self.F = utils.pad_array(F, (n_pop, F.shape[1]))

    def _evaluate_return(self, weights):
        raise NotImplementedError

    def optimize_return(self, *args, **kwargs):
        # private method
        # used inside parrelellization

        self.optimize(*args, **kwargs)
        return self.weights, self.F
        
    def re_evaluate_weights_single(self, weights):
        # wrapper around evaluate_return that handles 1d array of weights
        # evalauting a single portfolio choice

        # reshaping to be 2d
        weights = weights.reshape(1, -1)

        # re-valuating weight averages
        F = np.array(self._evaluate_return(weights)).T

        # post processed weights and metrics
        # ensuring that constraints are satisfied
        weights, F = self.post_process(weights, F)

        # returning weights to 1d
        return weights[0], F

    def re_evaluate_weights_vector(self, weights):
        F = np.array(self._evaluate_return(weights)).T
        weights, F = self.post_process(weights, F)

        return weights, F



    



class PortfolioProblem(BasePortfolioProblem):
    
    
    def __init__(self, returns, metrics, constraint_obj = constraints.LongOnly(), **kwargs):
        super().__init__(n_var=returns.shape[1], n_obj=len(metrics), 
                         metrics=metrics, constraint_obj = constraint_obj, **kwargs)

        # returns is a pandas DataFrame

        self.returns = returns.values
        self.column_names = returns.columns
        
    
    def _evaluate_return(self, weights):

        # enforces constraints before metrics can see
        weights = self.constraint_obj.enforce(weights)
        
        ret = (self.returns @ weights.T) if self.should_weight_returns else None

        # pymoo needs a list for some reason
        return [m.evaluate(ret, weights) for m in self.metrics]



class SubPortfolioProblem(BasePortfolioProblem):
    # portfolio problem where metrics are evaluated relative to the total portfolio
    # i.e., we are only changing the weights of the sub portfolio, but we want to optimize on the total portfolio impact
    
    def __init__(self, inner_returns, outer_returns, inner_w, metrics, constraint_obj = constraints.LongOnly(), **kwargs):
        super().__init__(n_var=inner_returns.shape[1], n_obj=len(metrics), 
                         metrics=metrics, **kwargs)

        # inner returns and outer returns need to be pd.DataFrames

        self.inner_returns = inner_returns.values
        self.column_names = inner_returns.columns
        self.outer_returns = (outer_returns.values * (1 - inner_w)).reshape(-1, 1)
        self.inner_w = inner_w


    def _evaluate_return(self, weights):

        # enforces constraints before metrics can see
        weights = self.constraint_obj.enforce(weights)

        # clean up code here
        weights_tot = self.inner_w * weights
        ret_tot = (self.inner_returns @ weights_tot.T + self.outer_returns) if self.should_weight_returns else None
        
        return [m.evaluate(ret_tot, weights) for m in self.metrics]



class MixedPortfolioProblem(BasePortfolioProblem):
    # portfolio Problem where some metrics are relative to total portfolio while others are not
    
    def __init__(self, inner_returns, outer_returns, inner_w, metrics, metric_levels, 
                 constraint_obj = constraints.LongOnly(), **kwargs):
        
        super().__init__(n_var=inner_returns.shape[1], n_obj=len(metrics), 
                         metrics=metrics, **kwargs)
        
        self.inner_returns = inner_returns.values
        self.column_names = inner_returns.columns
        self.outer_returns = (outer_returns.values * (1 - inner_w)).reshape(-1, 1)
        self.inner_w = inner_w
        self.metric_levels = metric_levels
        
    def _evaluate_return(self, weights):

        # enforces constraints before metrics can see
            
        # rename weights and inner weights, can be confusing
        weights_sub = self.constraint_obj.enforce(weights)
        weights_tot = self.inner_w * weights_sub
        
        if self.should_weight_returns:
            ret_sub = self.inner_returns @ weights_sub.T
            ret_tot = self.inner_returns @ weights_tot.T + self.outer_returns
        else:
            ret_sub = None
            ret_tot = None
    
            
        # level = 1 sub portfolio level, level = 0 is total portfolio
        weights_use = [((ret_sub, weights_sub) if level == 1 else (ret_tot, weights_tot)) for level in self.metric_levels]
        return [m.evaluate(r, w) for (r, w), m in zip(weights_use, self.metrics)]
    



    
