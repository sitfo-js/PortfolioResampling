import numpy as np
import pandas as pd
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import problem
from . import utils

# expand this to handle different ways of aggregating frontier
# redesign to take list or single object

class BaseResampler:
    def __init__(self, problem_func, data_args):

        problem_func, data_args = self.check_input(problem_func, data_args)
        self.problem_func = problem_func

        # check that stochastic and window resamplers aren't mixed
        self.data_args = data_args


        true_opt_obj = self.problem_func(*[arg.true_data for arg in data_args])

        # check that function indeed returns BasePortfolioProblem
        assert isinstance(true_opt_obj, problem.BasePortfolioProblem), "Must Return BasePortfolioProblem instance"

        true_opt_obj.optimize()

        self.true_opt_obj = true_opt_obj
        self.column_names = true_opt_obj.column_names
        self.metric_names = true_opt_obj.metric_names

        self.n_assets = len(self.column_names)
        self.n_metrics = len(self.metric_names)

        self.aggregated_weights = None
        self.aggregated_F = None




    def optimize(self, n_resamples = 100, n_pop = 100, n_gen = 100):

        # fix n_resamples to use window num rather than this num

        # potentially different population and generation parameters than true optimization

        self.n_resamples = n_resamples
        self.resampled_weights = np.empty(shape = (self.n_resamples, n_pop, self.n_assets))
        self.resampled_F = np.empty(shape = (self.n_resamples,  n_pop, self.n_metrics))


        futures = []
        with ProcessPoolExecutor() as executor:
            for n in range(n_resamples):
                resampled_args = [arg.resample() for arg in self.data_args]
                problem_obj = self.problem_func(*resampled_args)
                futures.append(executor.submit(problem_obj.optimize_return, n_pop = n_pop, n_gen = n_gen))


        for i, future in enumerate(futures):
            weights, f = future.result()

            # padding with nans in case optimizer can't find pareto optimal solutions as big as pop
            self.resampled_weights[i] = utils.pad_array(weights, (n_pop, self.n_assets))
            self.resampled_F[i] = utils.pad_array(f, (n_pop, self.n_metrics))

    def check_input(self, problem_func ,data_args):

        # private method
        if isinstance(data_args, Data_Wrapper):
            # wrapping in list
            data_args = [data_args]
        elif isinstance(data_args, list):
            # change for raising exception
            assert all([isinstance(arg, Data_Wrapper) for arg in data_args]), "Must pass Data_Wrapper or list of Data_Wrappers"


        return problem_func, data_args

    def aggregate_frontier(self):
        raise NotImplementedError

    def get_aggregate_frontier(self):
        # if frontier already formed, returns
        # else, forms frontier, then returns

        if (self.aggregated_F.size == 0) and (self.aggregated_weights.size == 0):
            self.aggregate_frontier()

        return self.aggregated_weights, self.aggregated_F


class ClusterResampler(BaseResampler):
    def __init__(self, problem_func, data_wrappers, n_clusters = 10):
        super().__init__(problem_func, data_wrappers)

        self.n_clusters = n_clusters
        self.aggregated_weights = np.empty((n_clusters, self.n_assets))
        self.aggregated_F = np.empty((n_clusters, self.n_metrics))

        self.cluster_model = None
        self.true_labels = np.empty(self.true_opt_obj.F.shape[0])

        # 1 sleeve holds true, other holds resampled
        # self.avg_w_cluster = np.empty((2, n_clusters, self.n_assets))


    def fit_cluster(self):
        # private method
        print("Hello 1")
        self.resampled_labels = np.empty((self.resampled_F.shape[0], self.resampled_F.shape[1]))

        true_F = self.true_opt_obj.F
        true_F_no_nan = utils.drop_nan_rows(true_F)

        # Create a pipeline with StandardScaler and KMeans
        pipeline = Pipeline([
            ("scaler", StandardScaler()),  # normalize the data
            ("kmeans", KMeans(n_clusters = self.n_clusters, n_init="auto"))
        ])


        true_labels_no_nan = pipeline.fit_predict(true_F_no_nan)
        true_labels = utils.pad_array(true_labels_no_nan, (true_F.shape[0],1)).squeeze()

        self.cluster_model = pipeline
        self.true_labels = true_labels

    def predict_resampled_labels(self):
        # private method
        print("Hello 2")
        for i in range(self.n_resamples):

            # looping through n_resamples
            # all nan rows will be at the end, so it won't mix indexing with weights
            iteration_F = utils.drop_nan_rows(self.resampled_F[i])

            predicted_labels = self.cluster_model.predict(iteration_F)
            print(predicted_labels)

            # saving, padding to fit original shape
            self.resampled_labels[i] = utils.pad_array(predicted_labels, (self.resampled_F.shape[1], 1)).squeeze()


    def aggregate_frontier(self):
        # public method

        # fitting cluster model on true
        self.fit_cluster()

        # predicting cluster of each resample iteration
        self.predict_resampled_labels()

        for j in range(self.n_clusters):
            indices_with_labels = self.resampled_labels == j

            # doesn't need nanmean because nan rows won't be labelled
            avg_weights = self.resampled_weights[indices_with_labels].mean(axis = 0)

            # can't simply avg F--we need to re-evalaute it in the true space
            # also, we need to put the average weights through processing to ensure constraints are satisfied

            avg_w, eval_F = self.true_opt_obj._evaluate_return_single(avg_weights)
            self.aggregated_weights[j] = avg_weights
            self.aggregated_F[j] = eval_F




class RankResampler(BaseResampler):
    def __init__(self, problem_func, data_wrappers, sort_col = 0):
        super().__init__(problem_func, data_wrappers)

        # check if problem is 2d

        self.sort_col = sort_col
        self.aggregated_weights = np.empty_like(self.true_opt_obj.weights)
        self.aggregated_F = np.empty_like(self.true_opt_obj.F)

    def aggregate_frontier(self):
        # public method

        sorted_weights, sorted_F = utils.sort_X_F_3d(self.resampled_weights, self.resampled_F)

        avg_weights = np.nanmean(sorted_weights, axis = 0)

        # re-evaluating resampled weights in the true space, enforcing constraints

        self.aggregated_weights, self.aggregated_F = self.true_opt_obj.re_evaluate_weights_vector(avg_weights)









class Data_Wrapper:
    def __init__(self, data):
        self.true_data = data

    def resample(self):
        raise NotImplementedError

class Constant(Data_Wrapper):
    def __init__(self, data):
        super().__init__(data)

    def resample(self):
        return self.true_data


class WindowResampler(Data_Wrapper):
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
            yield self.true_data.iloc[start:i]


class StochasticResampler(Data_Wrapper):
    def __init__(self, data, seed = None):
        self.gen = np.random.default_rng(seed = seed)
        super().__init__(data)

class BootstrapResampler(StochasticResampler):
    def __init__(self, data, seed = None):
        self.T = data.shape[0]
        super().__init__(data, seed)


class IIDBootstrap(BootstrapResampler):
    def __init__(self, data, seed = None):
        super().__init__(data, seed)

    def resample(self):
        indices = self.gen.integers(0, self.T, self.T)
        return self.true_data.iloc[indices]

# check that this works
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
        return self.true_data.iloc[indices]


# check that these work
class WhiteNoise(StochasticResampler):
    def __init__(self, data, seed = None):
        super().__init__(data, seed)

class GuassianNoise(WhiteNoise):
    def __init__(self, data, sigma, seed = None):
        self.sigma = sigma

        super().__init__(data, seed)

    def resample(self):
        return self.true_data + self.gen.normal(0, self.sigma, self.true_data.shape)


class CustomNoiseDist(WhiteNoise):
    def __init__(self, data, dist_func, *dist_args, seed = None):
        self.dist_func = dist_func
        self.dist_args = dist_args

        super().__init__(data, seed)

    def resample(self):
        return self.true_data + self.dist_func(self.gen, *self.dist_args)









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
        perturbed_weights[i] = utils.pad_array(weights, perturbed_weights[0].shape)
        perturbed_F[i] = utils.pad_array(f, perturbed_F[0].shape)
    
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
        






        

#algorithm = SMSEMOA(pop_size = params["Population Size"])   


