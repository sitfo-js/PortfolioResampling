
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from resampling import pad_array
import numpy as np


def get_cluster_labels(model_results, n_clusters):
    F = model_results["True F"]
    pert_F = model_results["Perturbed F"]

    # Drop NaN values from F
    F_no_nan = F[~np.isnan(F).any(axis=1)]

    # Create a pipeline with StandardScaler and KMeans
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # normalize the data
        ("kmeans", KMeans(n_clusters=n_clusters, n_init="auto"))
    ])
    
    pipeline.fit(F_no_nan)
    true_labels_no_nan = pipeline.predict(F_no_nan)

    # Pad true labels with NaN values to match original shape
    true_labels = pad_array(true_labels_no_nan, (F.shape[0],1)).squeeze()

    perturbed_labels = np.empty((pert_F.shape[0], pert_F.shape[1]))

    for i, perturbed_f in enumerate(pert_F):
        # Drop NaN values from perturbed_f
        perturbed_f_no_nan = perturbed_f[~np.isnan(perturbed_f).any(axis=1)]

        perturbed_labels_no_nan = pipeline.predict(perturbed_f_no_nan)

        # Pad perturbed labels with NaN values to match original shape
        perturbed_labels[i] = pad_array(perturbed_labels_no_nan, (perturbed_f.shape[0],1)).squeeze()

    return {
        "True Labels": true_labels,
        "Perturbed Labels": perturbed_labels,
        "N Clusters": pipeline.named_steps['kmeans'].cluster_centers_.shape[0]
    }


def get_cluster_attr(model_results, cluster_results, cluster_num, agg_type = "Perturbed"):
    

    true_indices = cluster_results["True Labels"] == cluster_num
    true_metrics = model_results["True F"][true_indices]
    true_weights = model_results["True Weights"][true_indices] 
    
    
    pert_indices = cluster_results["Perturbed Labels"] == cluster_num
    pert_metrics = model_results["Perturbed F"][pert_indices]
    pert_weights = model_results["Perturbed Weights"][pert_indices]
        

        
    return {
        "True Indices": true_indices,
        "True Metrics": true_metrics,
        "True Weights": true_weights,
        
        "Perturbed Indices": pert_indices,
        "Perturbed Metrics": pert_metrics,
        "Perturbed Weights": pert_weights,
        
        }

def get_resampled_frontier_cluster(model_results, cluster_results, key = "Perturbed Weights"):
    
    n = cluster_results["N Clusters"]
    
    avg_allocations = np.empty((n, model_results["True Weights"].shape[1]))
    avg_metrics = np.empty((n, model_results["True F"].shape[1]))
    portfolio_obj = model_results["True Problem Object"]
    
    
    for cluster_num in range(cluster_results["N Clusters"]):
        cluster_attr = get_cluster_attr(model_results, cluster_results, cluster_num)
        
        # average weights
        weights = cluster_attr[key].mean(axis = 0)
        
        weights, metrics = manual_evaluate_weights(portfolio_obj, weights)
        
        
        avg_allocations[cluster_num] = weights[0]
        avg_metrics[cluster_num] = metrics
        
        
    return avg_allocations, avg_metrics
        

def manual_evaluate_weights(portfolio_obj, weights):
    
    if weights.ndim == 2:
        reshaped = weights
    else:
        reshaped = weights.reshape(1, -1)
    
    # re-valuating weight averages
    metrics = np.array(portfolio_obj._evaluate_return(reshaped)).T
    
    # post processed weights and metrics
    weights, metrics = portfolio_obj.post_process(reshaped, metrics)
    
    return weights, metrics