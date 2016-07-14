"""
Defs for sigclust, cluster_index2, MAD, comp_sim_var, comp_sim_tau.
"""

import numpy as np
from sklearn.cluster import k_means
from sklearn.preprocessing import scale as pp_scale


def sigclust(X, mc_iters=100, method=2, verbose=True, scale=True,
             p_op=True, ngrains=100):
    """
    Return tuple (p, clust) where p is the p-value for k-means++ clustering of
        data matrix X with k==2, and clust is a (binary) array of length
        num_samples = X.shape[0] whose Nth value is the cluster assigned to
        the Nth sample (row) of X at the k-means step.  
        Equivalently, clust == k_means(X,2)[1].

    mc_iters is an integer giving the number of iterations in
        the Monte Carlo step.

    method = 0 uses the sample covariance matrix eigenvalues directly
        for simulation.
    method = 1 applies hard thresholding.
    method = 2 applies soft thresholding.

    scale = True  applies mean centering and variance normalization
        (sigma = 1) preprocessing to the input X.
    verbose = True prints some additional statistics of input data.

    When method == 2 (solf-thresholding), p_op indicates to perform some
        addiitonal optimization on the parameter tau.  If p_op == False,
        the parameter tau is set to that which best preserves the trace of
        the sample covariance matrix (this is just the output of comp_sim_tau):
        sum_{i}lambda_i == sum_{i}({lambda_i - tau - sigma^2}_{+} + sigma^2).
        If p_op == True, tau is set to some value between 0 and the output
        of comp_sim_tau which maximizes the relative size of the largest
        simulation variance. ngrains is then number of values checked in
        this optimization.  p_op and ngrains are ignored for method != 2.
    """
    if scale:
        print("Scaling and centering input matrix.")
        X = pp_scale(X)
    num_samples, num_features = X.shape
    if verbose:
        print("""Number of samples: %d\nNumber of features: %d""" %
              (num_samples, num_features))

    ci, labels = cluster_index_2(X)
    print("Cluster index of input data: %f" % ci)

    mad = MAD(X)
    if verbose:
        print("Median absolute deviation from the median of input data: %f"
              % mad)

    bg_noise_var = (mad * normalizer) ** 2
    print("Estimated variance for background noise: %f"
          % bg_noise_var)

    sample_cov_mat = np.cov(X.T)

    eig_vals = np.linalg.eigvals(sample_cov_mat)

    sim_vars = comp_sim_vars(eig_vals, bg_noise_var, method, p_op, ngrains)

    if verbose:
        print("The %d variances for simulation have\nmean: %f\n"
              "standard deviation: %f."
              % (X.shape[1], np.mean(sim_vars), np.std(sim_vars)))

    sim_cov_mat = np.diag(sim_vars)

    # MONTE CARLO STEP

    # Counter for simulated cluster indices less than or equal to ci.
    lte = 0
    print("""Simulating %d cluster indices.  Please wait...""" %
          mc_iters)
    CIs = np.zeros(mc_iters)
    for i in np.arange(mc_iters):
        # Generate mc_iters datasets
        # each of the same size as
        # the original input.
        sim_data = np.random.multivariate_normal(
            np.zeros(num_features), sim_cov_mat, num_samples)
        ci_sim = (cluster_index_2(sim_data))[0]
        CIs[i] = ci_sim
        if ci_sim <= ci:
            lte += 1
    print("Simulation complete.")
    print("The simulated cluster indices had\n"
          "mean: %f\nstandard deviation: %f." %
          (np.mean(CIs), np.std(CIs)))

    p = lte / mc_iters
    print("In %d iterations there were\n"
          "%d cluster indices <= the cluster index %f\n"
          "of the input data." % (mc_iters, lte, ci))
    print("p-value:  %f" % p)
    return p, labels

def cluster_index_2(X):
    global_mean = np.mean(X, axis=0)

    sum_squared_distances = (((X - global_mean) ** 2).sum(axis=1)).sum()
    # Sum of squared distances of each sample from the global mean

    centroids, labels, inertia = k_means(X, 2)

    ci = inertia / sum_squared_distances

    return ci, labels


normalizer = 1.48257969
"""
Equal to 1/(Phi^{-1}(3/4)) where Phi is the CDF
of the standard normal distribution N(0, 1)
"""

def MAD(X):
    """
    Returns the median absolute deviation
    from the median for (a flattened version of)
    the input data array X.
    """
    return np.median(np.abs(X - np.median(X)))


def comp_sim_vars(eig_vals, bg_noise_var, method, p_op, ngrains):
    """
    Compute variances for simlulation given sample variances and
        background noise variance.
    method in {0, 1, 2}  determines raw, hard, or soft thresholding methods.
    When method  is 2 (solf-thresholding), p_op indicates to perform some
        addiitonal optimization on the parameter tau.
    If p_op is False, the parameter tau is set to that which best preserves
        the trace of the sample covariance matrix.
        This is just the output of comp_sim_tau.
    sum_{i}lambda_i == sum_{i}{lambda_i - tau - sigma^2}_{+} + sigma^2).
    If p_op is True, tau is set to some value between 0 and the output
        of comp_sim_tau which maximizes the relative size of the largest
        simulation variance.
    """

    rev_sorted_vals = np.array(sorted(eig_vals, reverse = True))

    assert method in {0, 1, 2}, "method parameter must be one of 0,1,2"
    if method == 0:
        print("Ignoring background noise and using\n"
              "raw sample covariance estimates...")
        return rev_sorted_vals
    elif method == 1:
        print("Applying hard thresholding...")
        return np.maximum(rev_sorted_vals,
                          bg_noise_var * np.ones(len(eig_vals)))
    else:
        print("Applying soft thresholding...")
        tau = comp_sim_tau(rev_sorted_vals, bg_noise_var, p_op, ngrains)
        sim_vars = rev_sorted_vals - tau
        return np.maximum(sim_vars,
                          bg_noise_var * np.ones(len(eig_vals)))


def comp_sim_tau(rsrtd_vals, bg_noise_var, p_op, ngrains):
    """
    Find tau to preserve trace of sample cov matrix in the simulation
        cov matrix.

    sum_{i}(lambda_i) = sum_{i}{(lambda_i - tau - bg_noise_var)_{+} + bg_noise_var}
    """
    diffs = rsrtd_vals - bg_noise_var
    first_nonpos_ind = len(diffs) - len(diffs[diffs <=0])
    expended = -diffs[first_nonpos_ind:].sum()
    possible_returns = np.sort(diffs[:first_nonpos_ind]).cumsum()[::-1]
    tau_bonuses = np.arange(first_nonpos_ind) * diffs[:first_nonpos_ind]
    deficits = expended - possible_returns - tau_bonuses
    pos_defic = deficits > 0

    if pos_defic.sum() == 0:
        tau = expended / first_nonpos_ind
    else:
        ind = np.where(pos_defic)[0][0]
        if ind == 0:
            tau = diffs[0]
        else:
            tau = diffs[ind] + deficits[ind] / ind

    if p_op:
        tau = opt_tau(0, tau, rsrtd_vals, bg_noise_var, ngrains)
    return tau


def opt_tau(L, U, vals, sig2, ngrains=100):
    """Find tau between L and U that optimizes TCI."""
    d_tau = (U - L)/ngrains
    rel_sizes = np.zeros(ngrains)
    for i in np.arange(ngrains):
        tau_cand = L + i * d_tau
        vals0 = vals - tau_cand
        vals0[vals0 < sig2] = sig2
        rel_sizes[i] = vals0[0] / vals0.sum()
    armax = np.argmax(rel_sizes)
    tau = L + armax * d_tau

    return tau
