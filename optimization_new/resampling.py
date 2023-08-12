import numpy as np
import pandas as pd
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.optimize import minimize
from concurrent.futures import ProcessPoolExecutor


class Resampler:
    def __init__(self, problem_func, data_args = []):
        self.problem_func = problem_func
        self.data_args = data_args


        true_opt_obj = self.problem_func(*[arg.true for arg in data_args])
        true_opt_obj.optimize()

        self.true_opt_obj = true_opt_obj
        self.n_assets = self.true_opt_obj.weights.shape[1]
        self.n_metrics = self.true_opt_obj.F.shape[1]

    def resample(self, n_resamples = 100, n_pop = 100, n_gen = 100):
        # potentially different population and generation parameters than true optimization

        self.n_resamples = n_resamples
        self.resampled_weights = np.empty(shape = (self.n_resamples, n_pop, self.n_assets))
        self.resampled_F = np.empty(shape = (self.n_resamples,  n_pop, self.n_metrics))


        futures = []
        with ProcessPoolExecutor as executor:
            for n in n_resamples:
                resampled_args = [arg.resample() for arg in self.data_args]
                problem_obj = self.problem_func(*resampled_args)
                futures.append(executor.submit(problem_obj.optimize_return, n_pop = n_pop, n_gen = n_gen))


        for i, future in enumerate(futures):
            weights, f = future.result()
            self.resampled_weights[i] = pad_array(weights, (n_pop, self.n_assets))
            self.resampled_F[i] = pad_array(f, (n_pop, self.n_metrics))




class Resample_Wrapper:
    def __init__(self, data):
        self.true_data = data

    def resample(self):
        raise NotImplementedError

class Constant(Resample_Wrapper):
    def __init__(self, data):
        self.true_data = data

    def resample(self):
        return self.true_data


class WindowResampler(Resample_Wrapper):
    def __init__(self, data, min_periods = None, gap_size = 1, ascending = True):
        # X is np array or dataframe indexed by time across first dimension

        self.T = data.shape[0]
        self.min_periods = min_periods
        self.gap_size = gap_size
        self.ascending = ascending
        self.window_indices = []

        self.calculated_n_resamples = len([_ for _ in self.resample_generator()])
        self.gen = self.resample_generator()

        super().__init__(data)

    def resample_generator(self):
        raise NotImplementedError

    def resample(self):
        return next(self.gen)

    def get_start_end(self):
        start = self.min_periods if self.ascending else self.T
        end = self.T if self.ascending else self.min_periods
        gap_size = self.gap_size if self.ascending else -self.gap_size

        return start, end, gap_size



class RollingWindow(WindowResampler):
    def __init__(self, data, window = 12, min_periods = None, gap_size = 1, ascending = True):
        # assume data is pandas dataframe

        self.window = window

        super().__init__(data, min_periods, gap_size, ascending)

    def resample_generator(self):

        # flipping to move descendingly through the data
        start, end, gap_size = self.get_start_end()

        # setting to empty list again in case this gets called twice
        self.window_indices = []

        for i in range(start, end, gap_size):
            j = i - self.window


            # assumes argument is dataframe which needs to be abstracted
            self.window_indices.append([j, i])
            yield self.true_data.iloc[j:i]


class ExpandingWindow(WindowResampler):
    def __init__(self, data, min_periods = None, gap_size = 1, ascending = True):
        # assume X is pandas dataframe
        super().__init__(data, min_periods, gap_size, ascending)

    def resample_generator(self):

        # flipping to move descendingly through the data
        start, end, gap_size = self.get_start_end()

        # setting to empty list again in case this gets called twice
        self.window_indices = []

        for i in range(start, end, gap_size):
            self.window_indices.append([start, i])
            yield self.true_data[start:i]


class StochasticResampler(Resample_Wrapper):
    def __init__(self, seed = None):
        self.gen = np.random.default_rng(seed = seed)


class BootstrapResampler(StochasticResampler):
    def __init__(self, data, seed = None):
        self.T = data.shape[0]

        super().__init__(data, seed)


class IIDBootstrap(BootstrapResampler):
    def __init__(self, data, seed = None):
        super().__init__(data, seed)

    def resample(self):
        indices = self.gen.integers(0, self.T, self.T)
        return self.true_data[indices]

class BlockBootstrap(BootstrapResampler):
    def __init__(self, data, avg_block_size = 4, seed = None):
        self.avg_block_size = avg_block_size
        super().__init__(data, seed)

    def resample(self):
        indices = []

        # forward wrapping
        while len(indices) < self.T:
            idx = self.gen.integers(0, self.T)
            block_size = self.gen.geometric(1 / self.avg_block_size)
            indices.extend([(idx + x) % self.T for x in range(block_size)])

        indices = indices[:self.T]
        return self.true_data[indices]

class WhiteNoise(StochasticResampler):
    def __init__(self, data, seed = None):
        super().__init__(data, seed)

class GuassianNoise(WhiteNoise):
    def __init__(self, data, sigma, mu = 0, seed = None):
        self.mu = mu
        self.sigma = sigma

        super().__init__(data, seed)

    def resample(self):
        return self.true_data + self.gen.normal(self.mu, self.sigma, self.true_data.shape)


class CustomNoiseDist(WhiteNoise):
    def __init__(self, data, dist_func, *dist_args, seed = None):
        self.dist_func = dist_func
        self.dist_args = dist_args

        super().__init__(data, seed)

    def resample(self):
        return self.true_data + self.dist_func(self.gen, *self.dist_args)













class BaseResampler:
    def __init__(self, problem_func, args_to_resample = [], args_held_constant = []):
        
        self.problem_func = problem_func
        self.args_to_resample = args_to_resample
        self.args_held_constant = args_held_constant
        
        true_opt_obj = self.problem_func(*args_to_resample, *args_held_constant)
        true_opt_obj.optimize()
            
        self.true_opt_obj = true_opt_obj
        self.n_assets = self.true_opt_obj.weights.shape[1]
        self.n_metrics = self.true_opt_obj.F.shape[1]
            
        
        

        
    def resample(self, n_resamples = 100, n_pop = 100, n_gen = 100):
        # potentially different population and generation parameters than true optimization
        
        self.n_resamples = n_resamples
        self.resampled_weights = np.empty(shape = (self.n_resamples, n_pop, self.n_assets))
        self.resampled_F = np.empty(shape = (self.n_resamples,  n_pop, self.n_metrics))
        
        
        futures = []
        with ProcessPoolExecutor() as executor:
            for resampled_args in self.get_resample_args():
                problem_obj = self.problem_func(*resampled_args, *self.args_held_constant)
                futures.append(executor.submit(problem_obj.optimize_return, n_pop = n_pop, n_gen = n_gen))
            
        
        for i, future in enumerate(futures):
            weights, f = future.result()
            self.resampled_weights[i] = pad_array(weights, (n_pop, self.n_assets))
            self.resampled_F[i] = pad_array(f, (n_pop, self.n_metrics))
        
            
            
    def get_resample_args(self):
        # generator to yield resampled datapoints
        # should be implemented by each sub class
        
        raise NotImplementedError
        
        
# add checks that all arguments should have the same length



class WindowResampler(BaseResampler):
    def __init__(self, problem_func, args_to_resample = [], args_held_constant = [], 
                 min_periods = None, gap_size = 1, ascending = True):
        
        self.min_periods = min_periods
        self.ascending = ascending
        self.gap_size = gap_size
        self.window_indices = []
        
        # assumes all arguments have the same shape and that they are time indexable
        self.T = args_to_resample[0].shape[0]
        
        super().__init__(problem_func, args_to_resample, args_held_constant)
        

    def resample(self, n_pop = 100, n_gen = 100):
        self.n_resamples = len([x for x in self.get_resample_args()])
        super().resample(n_resamples = self.n_resamples, n_pop = n_pop, n_gen = n_gen)


        

# handle min periods consistently? should I subtract min_periods from the end if not ascending?

class RollingWindowResampler(WindowResampler):
    def __init__(self, problem_func, args_to_resample = [], args_held_constant = [], window_size = 12, 
                 min_periods = None, gap_size = 1, ascending = True):
        
        
        
        if min_periods is None:
            min_periods = window_size
            
        self.window_size = window_size
        
        super().__init__(problem_func, args_to_resample, args_held_constant, min_periods, gap_size, ascending)
        
    def get_resample_args(self):
        
        # flipping to move descendingly through the data
        start = 0 if self.ascending else self.T
        end = self.T if self.ascending else 0
        gap_size = self.gap_size if self.ascending else -self.gap_size
        
        # setting to empty list again in case this gets called twice
        self.window_indices = []
        
        for i in range(start, end, gap_size):
            j = i - self.window_size
            
            if j < self.min_periods:
                continue
            
            # assumes argument is dataframe which needs to be abstracted
            self.window_indices.append([j, i])
            yield [arg.iloc[j:i] for arg in self.args_to_resample]
    
        
        
        
        
    
        
class ExpandingWindowResampler(BaseResampler):
    def __init__(self, problem_func, args_to_resample = [], args_held_constant = [], min_periods = None,
                 gap_size = 1, ascending = True):
        
        
        if min_periods is None:
            min_periods = 0

        
        super().__init__(problem_func, args_to_resample, args_held_constant, min_periods, gap_size, ascending)
        
    def get_resample_args(self):
        
        start = self.min_periods if self.ascending else self.T
        end = self.T if self.ascending else self.min_periods
        gap_size = self.gap_size if self.ascending else -self.gap_size
        
        for i in range(start, end, gap_size):
            # expanding windows of the data
            yield [arg.iloc[start:i] for arg in self.args_to_resample]
            
        
    def resample(self, n_pop = 100, n_gen = 100):
        self.n_resamples = len([x for x in self.get_resample_args()])
        super().resample(n_resamples = self.n_resamples, n_pop = n_pop, n_gen = n_gen)
    
        
        
class BootstrapResampler(BaseResampler):
    def __init__(self, problem_func, args_to_resample = [], args_held_constant = [], 
                 kind = "iid", avg_block_size = None, seed = None, n_resamples = 100):
        
        
        self.seed = seed
        self.avg_block_size = avg_block_size
        self.n_resamples = n_resamples
        
        self.gen = np.random.default_rng(seed = seed)
        
        # assumes all arguments have the same shape and that they are time indexable
        self.T = args_to_resample[0].shape[0]
        
        
        if kind == "iid":
            self.resampled_indices = self.iid_bootstrap()
        elif kind == "block":
            self.resampled_indices = self.block_bootstrap()
            
            
        # initializing parent class
        super().__init__(problem_func, args_to_resample, args_held_constant)

    def iid_bootstrap(self):
        return self.gen.integers(0, self.T, size = (self.T, self.n_resamples))

    
    def block_bootstrap(self):
        
        
        iid_starting_indices = self.iid_bootstrap()
        
        # corresponding random block_sizes, not all will be used
        block_sizes = self.gen.geometric(1 / self.avg_block_size, size = iid_starting_indices.shape)
      
        
        # to return
        resampled_indices = np.empty((self.T, self.n_resamples))
        
        
        wrap_func = lambda idx, adj, T: idx - adj if (idx - adj) > 0 else T - adj
        
        for k in range(iid_starting_indices.shape[1]):
            
            col = []
            n = 0
            
            while n < self.T:
                start_idx = iid_starting_indices[n, k]
                block_size = block_sizes[n, k]
                block = [wrap_func(start_idx, x, self.T) for x in range(block_size)]
                
                col += block
                n += len(block)
                
            
            resampled_indices[:, k] = col[:self.T]
        
        return resampled_indices
    
    def get_resample_args(self):
        
        for indices in self.resampled_indices:
            yield [arg.iloc[indices] for arg in self.resampled_args]
            
    



class PerturbResampler(BaseResampler):
    def __init__(self, problem_func, args_to_resample = [], args_held_constant = [], 
                 sigmas = None, seed = None, custom_perturb_func = None):
        
        # raise error if custom func is not passed and sigmas are not passed
        
        self.sigmas = sigmas
        self.gen = np.random.default_rng(seed = seed)
        self.perturb_func = self.default_perturb_func if custom_perturb_func is None else custom_perturb_func
        
    def default_perturb_func(self, args, gen):
        return [arg + gen.normal(0, sigma) for arg, sigma in zip(args, self.sigmas)]
    
    def get_resample_args(self):
        for n in range(self.n_resamples):
            yield self.perturb_func(self.args_to_resample, self.gen)
        
        



        


            
            
        
        
        









def optimize(problem_obj, params, *args, algorithm = DNSGA2, true_model = True):
    
    if true_model:
        n_pop = params["Population Size True"]
        n_gen = params["Num Generations True"]
    else:
        n_pop = params["Population Size Perturbed"]
        n_gen = params["Num Generations Perturbed"]
        
        
    # initializes algorithm
    algo_instance = algorithm(pop_size = n_pop)
    
    # results
    res = minimize(problem_obj, algo_instance, 
                    ("n_gen", n_gen), seed = params["Seed"], 
                    verbose=False)
    
    # extracts and evaluated metrics and post processes
    return problem_obj.post_process(*res.opt.get("X", "F"))

def run_perturb(problem_func, params, shock_args, other_args):
    # iteratively perturbs data and runs model
    # uses parrellel processing for efficiency
    
    # "true" weights and results from history without perturbing data
    true_problem_obj = problem_func(params, *shock_args, *other_args)
    true_weights, true_F = optimize(true_problem_obj, params, *shock_args, *other_args, true_model = True)
    
    iterator = shocks_iterator(params, shock_args)
    shocked_args = [[s_arg + s for s_arg, s in zip(shock_args, shocks)] for shocks in iterator]
    
    # submitting optimization jobs
    futures = []
    with ProcessPoolExecutor() as executor:
        for shocked in shocked_args:
            problem_object = problem_func(params, *shocked, *other_args)
            futures.append(executor.submit(optimize, problem_object, params, 
                                           *shocked, *other_args, true_model = False))
        
    n = params["Num Variations"]
    pop_size = params["Population Size Perturbed"]
    
    perturbed_weights = np.empty(shape = (n, pop_size, true_weights.shape[1]))
    perturbed_F = np.empty(shape = (n, pop_size, true_F.shape[1]))
    
    for i, future in enumerate(futures):
        weights, f = future.result()
        perturbed_weights[i] = pad_array(weights, perturbed_weights[0].shape)
        perturbed_F[i] = pad_array(f, perturbed_F[0].shape)
    
    return {"Perturbed Weights": perturbed_weights,
            "Perturbed F": perturbed_F,
            "True Args": (*shock_args, *other_args),
            "Shocked Args": shocked_args,
            "True Weights": true_weights,
            "True F": true_F,
            "True Problem Object": true_problem_obj,
            "Column Names": shock_args[0].columns,
            "Metric Names": [m.name for m in true_problem_obj.metrics],
            "Params": params}


def run_optimize(problem_func, params, args):
    
    # running a single optimization
    
    true_problem_obj = problem_func(params, *args)
    true_weights, true_F = optimize(true_problem_obj, params, *args, true_model = True)
    
    
    return {
        "True Weights": true_weights,
        "True F": true_F,
        "True Problem Object": true_problem_obj,
        "Column Names": args[0].columns,
        "Metric Names": [m.name for m in true_problem_obj.metrics],
        "Params": params,
        }

    
    


def shocks_iterator(params, shock_args):
    
    gen = np.random.default_rng(seed = params["Seed"])
    for i in range(params["Num Variations"]):
        to_yield = [gen.normal(loc = 0, scale = sigma, size = shock_arg.shape) for shock_arg, sigma in zip(shock_args, params["Shock Sigmas"])]
        yield to_yield
        
        

def pad_array(array, target_shape):
    if len(array.shape) == 1:  # if it's 1D, reshape to 2D
        array = array.reshape(-1, 1)
    padded_array = np.full(target_shape, np.nan)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array





        

#algorithm = SMSEMOA(pop_size = params["Population Size"])   


