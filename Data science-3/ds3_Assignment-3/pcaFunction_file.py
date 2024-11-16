#pca
import numpy as np

def pca(data, n_components=1):
    data_meaned = data - np.mean(data, axis=0)
    covariance_matrix = np.cov(data_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    eigenvector_max = sorted_eigenvectors[:, :n_components]
    data_reduced = np.dot(data_meaned, eigenvector_max)
    
    return data_reduced


