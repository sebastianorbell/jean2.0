import numpy as np
from scipy.ndimage import gaussian_filter1d
def high_pass_and_normalise(array, f_min=None):
    array = array - gaussian_filter1d(array, sigma=3)
    array = (array - np.mean(array)) / np.std(array)
    return array

def PCA(*arrays):
    arrays_copy = [np.copy(array) for array in arrays]
    for array in arrays_copy:
        array[np.isnan(array)] = np.mean(np.ma.masked_invalid(array))
        array = high_pass_and_normalise(array)

    Z = np.stack(arrays_copy, axis=-1)
    shape = Z.shape

    # summing over every axis except the last
    u = np.mean(Z, axis=tuple(range(0, shape.__len__() - 1)), keepdims=True)

    B = (Z - u).reshape(np.product(shape[0:-1]), shape[-1])
    C = np.einsum("ki, kj -> ij", B, B)
    eigen_values, eigen_vectors = np.linalg.eig(C)
    arg_sorted = np.flip(eigen_values.argsort())
    eigen_vectors = eigen_vectors[:, arg_sorted]
    return np.einsum("ik, kj -> ij", B, eigen_vectors).reshape(shape)[..., 0]