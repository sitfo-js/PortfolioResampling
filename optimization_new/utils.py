import numpy as np

# change notation from all_X

def sort_X_F_3d(resampled_weights, resampled_F, sort_col = 0):

    sorted_X = np.empty_like(resampled_weights)
    sorted_F = np.empty_like(resampled_F)

    for i in range(resampled_F.shape[0]):
        sort_indices = np.argsort(resampled_F[i][:, sort_col])
        sorted_F[i] = resampled_F[i][sort_indices]
        sorted_X[i] = resampled_weights[i][sort_indices]

    return sorted_X, sorted_X

def sort_X_F_2d(weights, F, sort_col = 0):
    sort_indices = np.argsort(F[:, sort_col])
    return weights[sort_indices], F[sort_indices]

def drop_nan_rows(arr):
    return arr[~np.isnan(arr).any(axis=1)]

def pad_array(arr, target_shape):
    if len(arr.shape) == 1:  # if it's 1D, reshape to 2D
        array = arr.reshape(-1, 1)

    padded_array = np.full(target_shape, np.nan)
    padded_array[:arr.shape[0], :arr.shape[1]] = arr
    return padded_array