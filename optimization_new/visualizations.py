import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from . import resampling



def frontier_scatter(model_results, cluster_results = None, resampled_metrics = None, plot_metrics = None, plot_metrics_adj = None, plot_names = None,
                     scatter_type = None, c_idx = None, cmap = "viridis", view_angle = None):
    
    F = model_results["True F"]
    weights = model_results["True Weights"]
    
    ndim = F.shape[1]
    
    if (ndim == 2) or (scatter_type == "Color 3d"):
        f, ax = plt.subplots()
        
        args = (F[:, 0], F[:, 1])
        
        if scatter_type is None:
            kwargs = {}
            colorbar_name = None
            
        elif scatter_type == "Color 3d":
            kwargs = {"c": F[:, 2], "cmap":  cmap}
            colorbar_name = model_results["Metric Names"][2]
            
        elif scatter_type == "Color W":
            kwargs = {"c": weights[:, c_idx], "cmap": cmap}
            colorbar_name = model_results["Column Names"][c_idx]
            
        elif scatter_type == "Clusters":
            kwargs = {"c": cluster_results["True Labels"]}
            colorbar_name = None
        elif scatter_type == "Resampled":
            kwargs = {"color": "tab:blue", "label": "True Efficient Frontier"}
            colorbar_name = None
            ax.scatter(resampled_metrics[:, 0], resampled_metrics[:, 1], label = "Resampled Frontier", color = "tab:orange")
            
        elif scatter_type == "Proposed":
            kwargs = {"color": "tab:blue", "label": "Efficient Frontier"}
            colorbar_name = None
            colors = ["tab:red", "tab:brown","tab:cyan", "tab:purple"]
            
            for i, name in enumerate(plot_names):
                metrics = plot_metrics[i]
                ax.scatter(metrics[0], metrics[1], marker = "X", s = 150, label = name, color = colors[i], zorder = 2)
                
        elif scatter_type == "Proposed Resampled":
            kwargs = {"color": "tab:blue", "label": "True Efficient Frontier"}
            colorbar_name = None
            ax.scatter(resampled_metrics[:, 0], resampled_metrics[:, 1], label = "Resampled Frontier", color = "tab:orange")
            
            colors = ["tab:red", "tab:brown","tab:cyan", "tab:purple"]
            
            for i, name in enumerate(plot_names):
                metrics = plot_metrics[i]
                ax.scatter(metrics[0], metrics[1], marker = "X", s = 150, label = name, color = colors[i], zorder = 2)

            
        
        s = ax.scatter(*args, **kwargs)
        
        if scatter_type == "Resampled" or scatter_type == "Proposed" or scatter_type == "Proposed Resampled":
            ax.legend()
        
        if colorbar_name is not None:
            colorbar = plt.colorbar(s)
            colorbar.set_label(colorbar_name)
    
        ax.set_xlabel(model_results["Metric Names"][0])
        ax.set_ylabel(model_results["Metric Names"][1])
        ax.set_title("Efficient Frontier")
        
    #elif scatter_type == "Color 3d":
        #f, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    
    
    

def scatter_2d(model_results, cluster_results = None, cluster_num = None):
    F = model_results["True F"]
    
    f, ax = plt.subplots()
    alpha = 1 if cluster_num is None else 0.1
    ax.scatter(F[:, 0], F[:, 1], alpha = alpha)
    
    metric_names = model_results["Metric Names"]
    ax.set_xlabel(metric_names[0])
    ax.set_ylabel(metric_names[1])
    ax.set_title("Efficient Frontier")
    
    if cluster_results is not None:
        cluster_attr = resampling.get_cluster_attr(model_results, cluster_results, cluster_num)
        ax.scatter(cluster_attr["True Metrics"][:, 0], cluster_attr["True Metrics"][:, 1], color = "tab:orange")

    
def scatter_2d_color(model_results):
    F = model_results["True F"]
    
    f, ax = plt.subplots()
    s = ax.scatter(F[:, 0], F[:, 1], c = F[:, 2])
    
    metric_names = model_results["Metric Names"]
    ax.set_xlabel(metric_names[0])
    ax.set_ylabel(metric_names[1])
    ax.set_title("Efficient Frontier")
    
    colorbar = plt.colorbar(s)
    colorbar.set_label(metric_names[2])
    
def scatter_3d(model_results, view_angle = None, alpha = 1, c = None):
    
    
    F = model_results["True F"]
    
    f, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if c is None:
        ax.scatter(F[:, 0], F[:, 1], F[:, 2])
    else:
        s = ax.scatter(F[:, 0], F[:, 1], F[:, 2], c = model_results["True Weights"][:, c])
        colorbar = plt.colorbar(s)
        colorbar.set_label(model_results["Column Names"][c])
    
    metric_names = model_results["Metric Names"]
    ax.set_xlabel(metric_names[0])
    ax.set_ylabel(metric_names[1])
    ax.set_zlabel(metric_names[2])
    ax.set_title("Efficient Frontier")
    
    if view_angle is not None:
        ax.view_init(elev = int(view_angle[0]), azim = int(view_angle[1]))
        
def scatter_3d_clusters(model_results, cluster_results, view_angle = None, alpha = 1):
    
    F = model_results["True F"]
    labels = cluster_results["True Labels"]
    
    f, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c = labels, alpha = alpha)
    
    metric_names = model_results["Metric Names"]
    ax.set_xlabel(metric_names[0])
    ax.set_ylabel(metric_names[1])
    ax.set_zlabel(metric_names[2])
    ax.set_title("Efficient Frontier")
    
    if view_angle is not None:
        ax.view_init(elev = int(view_angle[0]), azim = int(view_angle[1]))
        
        
        
def plot_cluster_dist(model_results, cluster_results, cluster_num):
    
    cluster_attr = resampling.get_cluster_attr(model_results, cluster_results, cluster_num)
    
    n_metrics = model_results["True F"].shape[1]
    fig = plt.figure(figsize = (10, 7))

    # Create a GridSpec object
    gs = gridspec.GridSpec(n_metrics, 2)

    # For each metric, create a subplot in the left column
    for i in range(n_metrics):
        ax = fig.add_subplot(gs[i, 0])
        ax.boxplot(cluster_attr["True Metrics"][:, i], vert = False, widths = 0.75)
        ax.set_xlim(model_results["True F"][:, i].min(), model_results["True F"][:, i].max())
        ax.set_yticks([1])
        ax.set_yticklabels([model_results["Metric Names"][i]])

    # Create a subplot in the right column
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.boxplot(cluster_attr["True Weights"], vert = True)
    ax2.set_xticks(range(1, model_results['True Weights'].shape[1] + 1))
    ax2.set_xticklabels(model_results["Column Names"])
    ax2.set_ylim(0, 1)

    #plt.tight_layout()
    fig.suptitle("Cluster {}".format(cluster_num))
    plt.show()
    
def less_than_filter(col, thresh):
    return lambda F: F[:, col] < thresh

def greater_than_filter(col, thresh):
    return lambda F: F[:, col] > thresh

def filter_plot_view(model_results, filter_funcs = []):
    F = model_results["True F"]
    weights = model_results["True Weights"]
    n_dim = F.shape[1]
    
    filters = [func(F) for func in filter_funcs]
    
    if len(filters) == 0:
        total_filter = np.ones(weights.shape[0], dtype = bool)
    elif len(filters) == 1:
        total_filter = filters[0]
    elif len(filters) > 1:
        filters_concat = np.vstack(filters).T
        total_filter = np.all(filters_concat, axis = 1)
    
    filtered_F = F[total_filter]
    filtered_weights = weights[total_filter]
    
    not_in_filter_F = F[~total_filter]

    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 7))
    ax1.boxplot(filtered_weights, vert = False)
    ax1.set_yticks(range(1, weights.shape[1] + 1))
    ax1.set_yticklabels(model_results["Column Names"])
    ax1.set_xlim(0, 1)
    
    ax2.scatter(not_in_filter_F[:, 0], not_in_filter_F[:, 1], color = "tab:gray", alpha = 0.1)
    
    if n_dim == 2:
        ax2.scatter(filtered_F[:, 0], filtered_F[:, 1])
    elif n_dim == 3:
        s = ax2.scatter(filtered_F[:, 0], filtered_F[:, 1], c = filtered_F[:, 2])
        colorbar = plt.colorbar(s)
        colorbar.set_label(model_results["Metric Names"][2])
    
    #ax2.grid(True, axis = "y", color = "black", lw = 0.5)
    
def get_cluster_stats(model_results, cluster_results):
    n_clusters = cluster_results["N Clusters"]
    
    true_metrics = []
    true_weights = []
    pert_metrics = []
    pert_weights = []
    
    for i in range(cluster_results["N Clusters"]):
        attr = resampling.get_cluster_attr(model_results, cluster_results, i)
        
        true_metrics.append(attr["True Metrics"].mean(axis = 0))
        true_weights.append(attr["True Weights"].mean(axis = 0))
        pert_metrics.append(attr["Perturbed Metrics"].mean(axis = 0))
        pert_weights.append(attr["Perturbed Weights"].mean(axis = 0))
        
        
    true_m_cols = [("True", x) for x in model_results["Metric Names"]]
    true_metrics = pd.DataFrame(true_metrics, index = range(n_clusters),
                         columns = pd.MultiIndex.from_tuples(true_m_cols))
    
    true_w_cols = [("True", x) for x in model_results["Column Names"]]
    true_weights = pd.DataFrame(true_weights, index = range(n_clusters),
                               columns = pd.MultiIndex.from_tuples(true_w_cols))
    
    pert_m_cols = [("Pert", x) for x in model_results["Metric Names"]]
    pert_metrics = pd.DataFrame(pert_metrics, index = range(n_clusters),
                               columns = pd.MultiIndex.from_tuples(pert_m_cols))
    
    pert_w_cols = [("Pert", x) for x in model_results["Column Names"]]
    pert_weights = pd.DataFrame(pert_weights, index = range(n_clusters),
                               columns = pd.MultiIndex.from_tuples(pert_w_cols))

    master_table = pd.concat([true_metrics, true_weights, pert_metrics, pert_weights], axis = 1)
    master_table = master_table.sort_values(by = true_m_cols[0], ascending = True)
    
    return (master_table.loc[:, true_m_cols], 
            master_table.loc[:, true_w_cols], 
            master_table.loc[:, pert_m_cols],
            master_table.loc[:, pert_w_cols],)

def visualize_dispersion_scatter(model_results, col, show_true = True):
    num = model_results["Column Names"].to_list().index(col)
    
    perturbed_X = model_results["Perturbed X"]
    perturbed_F = model_results["Perturbed F"]
    
    f, ax = plt.subplots()
    i = 0
    for X, F in zip(perturbed_X, perturbed_F):
        label = "Weights Perturbed" if i == 0 else None
        ax.scatter(F[:, 0], X[:, num], color = "tab:blue", alpha = 0.05, label = label)
        i += 1
        
    if show_true:
        true_X = model_results["True X"]
        true_F = model_results["True F"]
        
        sorted_X, sorted_F = sort_X_F_2d(true_X, true_F)
        ax.plot(sorted_F[:, 0], sorted_X[:, num], color = "tab:red", label = "True Weights", lw = 3)
        
    ax.legend()
    ax.set_ylabel("Weight")
    ax.set_xlabel("Bear Beta")
    ax.set_ylim(0, 1)
    ax.set_title(col)
    
def sort_X_F_3d(all_X, all_F, sort_col = 0):
    sorted_F = np.empty_like(all_F)
    sorted_X = np.empty_like(all_X)

    for i in range(all_F.shape[0]):
        sort_indices = np.argsort(all_F[i][:, sort_col])
        sorted_F[i] = all_F[i][sort_indices]
        sorted_X[i] = all_X[i][sort_indices]
        
    return sorted_X, sorted_X

def sort_X_F_2d(X, F, sort_col = 0):
    sort_indices = np.argsort(F[:, sort_col])
    return X[sort_indices], F[sort_indices]

def visualize_dispersion_scatter(model_results, col, show_true = True):
    num = model_results["Column Names"].to_list().index(col)
    
    perturbed_X = model_results["Perturbed Weights"]
    perturbed_F = model_results["Perturbed F"]
    
    f, ax = plt.subplots()
    i = 0
    for X, F in zip(perturbed_X, perturbed_F):
        label = "Weights Perturbed" if i == 0 else None
        ax.scatter(F[:, 0], X[:, num], color = "tab:blue", alpha = 0.05, label = label)
        i += 1
        
    if show_true:
        true_X = model_results["True Weights"]
        true_F = model_results["True F"]
        
        sorted_X, sorted_F = sort_X_F_2d(true_X, true_F)
        ax.plot(sorted_F[:, 0], sorted_X[:, num], color = "tab:red", label = "True Weights", lw = 3)
        
    ax.legend()
    ax.set_ylabel("Weight")
    ax.set_xlabel("Bear Beta")
    ax.set_ylim(0, 1)
    ax.set_title(col)
    

def plot_portfolio_mixes_2d(model_results, cluster_results = None, resample = False):
    
    if resample:
        weights, metrics = resampling.get_resampled_frontier_cluster(model_results, cluster_results)
    else:
        weights = model_results["True Weights"]
        metrics = model_results["True F"]
    
    weights, metrics = sort_X_F_2d(weights, metrics)
    weights_df = pd.DataFrame(weights, index = metrics[:, 0], columns = model_results["Column Names"])
    
    f, ax = plt.subplots()
    weights_df.plot.area(ax = ax)

    ax.legend(bbox_to_anchor = (1, 1))
    ax.set_title("Weight Mixes Across Efficient Frontier" + ("" if not resample else " (Resampled)"))
    ax.set_ylabel("Weight")
    ax.set_xlabel(model_results["Metric Names"][0])
    
    plt.show()

    

    
    
    
    
    
    
    
    