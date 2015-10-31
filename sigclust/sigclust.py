import numpy as np
from sklearn.cluster import k_means
from sklearn.preprocessing import scale




def sigclust(X, mc_iters=100, floor=0,
             verbose=True, scale = False):
    """
    Returns p-value for k-means++ clustering of array X with k==2.
    X has shape (num_samples, num_features).
    mc_iters is an integer giving the number of iterations in 
    the Monte Carlo step.
    floor is an optional minimum on simulation variances.

    """
    if scale:
        X = scale(X)
    num_samples, num_features = X.shape
    print("Number of samples: %d, Numer of features: %d" % (num_samples, num_features))
    
    ci, labels = cluster_index_2(X)
    print("Cluster index of input data: %f" % ci)

    mad = MAD(X)
    print("Median absolute deviation from the median of input data:  %f" % mad)

    bg_noise_var = (mad*normalizer)**2
    print("Estimated variance for background noise: %f" % bg_noise_var)

    floor_final = max(floor, bg_noise_var)

    data_cov_mat = np.cov(X.T)

    eig_vals, eig_vects = np.linalg.eig(data_cov_mat)

    args = np.argsort(eig_vals)

    rev_sorted_args = args[::-1]
    
    rev_sorted_vals = eig_vals[rev_sorted_args]

    new_vars = np.maximum(rev_sorted_vals, floor_final * np.ones(num_features))

    print("Variances for simulation are:", new_vars)

    sim_cov_mat = np.diag(new_vars)

    if verbose:
        input("Press enter to begin simulation.")
    
    ##MONTE CARLO STEP

    #Counter for simulated cluster indices less than or equal to ci.
    lte = 0
    
    for i in np.arange(mc_iters):
    #Generate mc_iters datasets each of the same size as the original input.
        sim_data = np.random.multivariate_normal(np.zeros(num_features), sim_cov_mat, num_samples)

        """
        Shape of sim_data was (mc_iters, num_samples, num_features)
        """

        ci_sim = (cluster_index_2(sim_data))[0]
        print("Cluster index of simulated data set number %d is %f" % (i, ci_sim))

        if ci_sim <= ci:
            lte += 1
    #P value
    p = lte / mc_iters
        
    return p, labels


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
    


normalizer = 1.48257969
"""
Equal to 1/(Phi^{-1}(3/4)) where Phi is the CDF
of the standard normal distribution N(0, 1)
"""


def MAD(X):
    """
    Returns the median absolute deviation from the median
    for a data array X.  If X has dimension greater than 1, 
    returns MAD of flattened array.
    """
    return np.median(np.abs(X - np.median(X)))



