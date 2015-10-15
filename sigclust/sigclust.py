import numpy as np
from sklearn.cluster import k_means


def sigclust(X, mc_iters=100):
    """
    Returns p-value for k-means++ clustering of array X with k==2.
    X has shape (num_samples, num_features).
    mc is an integer giving the number of iterations in 
    the Monte Carlo step.

    """
    num_samples, num_features = X.shape
    
    ci, labels = cluster_index_2(X)

    bg_noise_var = (MAD(X)*normalizer)**2

    data_cov_mat = np.cov(X.T)

    eig_vals, eig_vects = np.linalg.eig(data_cov_mat)

    eig_vals.sort()

    new_vars = np.maximum(eig_vals, bg_noise_var * np.ones(num_features))

    sim_cov_mat = np.diag(new_vars)

    sim_data = np.random.multivariate_normal(np.zeros(num_features), sim_cov_mat, (mc_iters, num_samples))
    """
    Shape of sim_data is (mc_iters, num_samples, num_features)
    """

    ci_sims = np.zeros(mc_iters)
    
    for i in np.arange(mc_iters):
        ci_sims[i] = cluster_index_2(sim_data[i,:,:])[0]

    p = compute_pvalue(ci, ci_sims)

    return p, labels



def compute_pvalue(i, sims):
    """
    Returns the propositon of sims less than or equal to i
    """
    lower_than = sims <= i

    num_lower_than = np.sum(lower_than.astype(int))

    return num_lower_than / sims.size



def cluster_index_2(X):

    """
    Returns the pair consisting of the 2-means cluster index
    for X and the corresponding labels
    """
    
    global_mean = X.mean(axis=0)

    sum_squared_distances = (((X - global_mean)**2).sum(axis = 1)).sum()
    #Sum of squared distances of each sample from the global mean
    
    centroids, labels, inertia = k_means(X, 2)

    ci = inertia / sum_squared_distances

    return ci , labels
    

def MAD(X):
    """
    Returns the median absolute deviation from the median
    for a data array X.  If X has dimention greater than 1, 
    returns MAD of flattened array.
    """
    return np.median(np.abs(X - np.median(X)))




normalizer = 1.48257969
"""
Equal to 1/(Phi^{-1}(3/4)) where Phi is the CDF
of the standard normal distribution N(0, 1)
"""
