from sys import exit
import numpy as np
from sklearn.cluster import k_means
from sklearn.preprocessing import scale as pp_scale

"""Defs for sigclust, cluster_index2, MAD, recclust, comp_sim_var, comp_sim_tau"""


def sigclust(X, mc_iters=100, method = 2,
             verbose=True, scale = True):
    """
    Returns tuple with first element the p-value for k-means++ clustering of array X with k==2.  The second element of the returned tuple is a (binary) array of length num_samples = X.shape[0] whose Nth value is the cluster assigned to the Nth sample (row) of X at the k-means step. (Equivalently, sigclust(X)[1] == k_means(X,2)[1])
    mc_iters is an integer giving the number of iterations in the Monte Carlo step.
    method = 0 uses sample cov matrix eigenvalues directly for simulation.  method = 1 applies hard threasholding, m =  applies soft thresholding. 
    scale = True  applies mean centering and variance normalization (sigma = 1) preprocessing to the input X.
    verbose = True prints some additional statistics of input data.
    """
    if scale:
        print("Scaling and centering input matrix.")
        X = pp_scale(X)
    num_samples, num_features = X.shape
    if verbose:
        print("""Number of samples: %d, 
        Number of features: %d""" %
              (num_samples, num_features))
    
    ci, labels = cluster_index_2(X)
    print("Cluster index of input data: %f" % ci)

    mad = MAD(X)
    if verbose:
        print("""Median absolute deviation 
from the median of input data:  %f""" % mad)
        

    bg_noise_var = (mad*normalizer)**2
    print("""Estimated variance for 
background noise: %f""" % bg_noise_var)

    sample_cov_mat = np.cov(X.T)

    eig_vals, eig_vects = np.linalg.eig(sample_cov_mat)

    sim_vars = comp_sim_vars(eig_vals, bg_noise_var, method)
    
    if verbose:
        print("""
        The %d variances for simulation have
        mean: %f 
        standard deviation: %f.""" %
        (X.shape[1],
         np.mean(sim_vars),
         np.std(sim_vars)))

    sim_cov_mat = np.diag(sim_vars)
    
    ##MONTE CARLO STEP

    #Counter for simulated cluster indices less than or equal to ci.
    lte = 0
    print("""Simulating %d cluster indices.
Please wait...""" %
          mc_iters)
    CIs = np.zeros(mc_iters)
    for i in np.arange(mc_iters):
    #Generate mc_iters datasets
        #each of the same size as
        #the original input.
        sim_data = np.random.multivariate_normal(np.zeros(num_features), sim_cov_mat, num_samples)
        ci_sim = (cluster_index_2(sim_data))[0]
        CIs[i] = ci_sim
        if ci_sim <= ci:
            lte += 1
    #P value
    print("Simulation complete.")
    print("""The simulated cluster indices had
mean: %f
standard deviation: %f.""" %
          (np.mean(CIs), np.std(CIs)))
    
    p = lte / mc_iters
    print("""In %d iterations there were 
%d cluster indices <= the cluster index %f
 of the input data.""" %
          (mc_iters, lte, ci))
    print("p-value:  %f" % p)
    return p, labels

def cluster_index_2(X):    
    global_mean = np.mean(X, axis=0)

    sum_squared_distances = (((X - global_mean)**2).sum(axis = 1)).sum()
    #Sum of squared distances of each sample from the global mean
    
    centroids, labels, inertia = k_means(X, 2)

    ci = inertia / sum_squared_distances

    return ci , labels

###
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


def recclust(X, threshold = .01,
             mc_iters = 100, verbose = True,
             prefix = "/", IDS = np.arange(0)):
    
    """
    Recursively applies sigclust to until all remaining clusters have sigclust p-value below threshold.
    Returns dictionary with entries having keys "prefix", "pval", "subclust0", "subclust1", and "ids"
    "prefix" is a string representation for a path to the root cluster consisting of all samples in the input X.
    "pval" is the p-value of cluster 'prefix'
    "subclust{0,1}" are dictionaries fo the same form as this, representing the two primary subclusters of the cluster 'prefix'.  Note that these may be None if pval greater than threshold.
    "ids" is a numpy array of keys to use to record the cluster elements themselves.  Note that this may be None if pval is greater than threshold.
    "tot" is the total number of samples in the cluster.
    """
    if IDS.shape[0] == 0 :
        IDS = np.arange(X.shape[0])
    assert IDS.shape[0] == X.shape[0], """Input data and tag list must have
    compatible dimensions (or tag list
    must be None).
    """
    
    data = {"prefix" : prefix, "pval" : None,
           "subclust0" : None,
            "subclust1" : None,
            "ids" : None, "tot" : 1}
    
    if X.shape[0] == 1:
        data["ids"] = IDS
        print("""Cluster %s has exactly
        one element.""" % prefix)
    else:
        p, clust = sigclust(X,
                            mc_iters = mc_iters,
                            verbose = verbose)
        print("""The p value for
        subcluster id %s is %f""" %
              (prefix, p))
        data["pval"] = p
        
        if p >= threshold:
            data["ids"] = IDS
        else:
            pref0 = prefix + "0"
            pref1 = prefix + "1"
            print("""Examining sub-clusters
            %s and %s""" % (pref0, pref1))
            data_0 = X[clust == 0, :]
            data_1 = X[clust == 1, :]
            print("""Computing RecClust data
            for first cluster.
            Please wait...""")
            dict0 = recclust(data_0,
                             prefix = prefix + "0",
                             IDS = IDS[clust == 0])
            print("""Computing Recclust data
            for second cluster.
            Please wait...""")            
            dict1 = recclust(data_1,
                             prefix = prefix + "1",
                             IDS = IDS[clust == 1])
            data["subclust0"] = dict0
            data["subclust1"] = dict1
            data["tot"] = dict0["tot"] + dict1["tot"]
        
    return data

def comp_sim_vars(eig_vals, bg_noise_var, method):

    #First sort eig_vals
    args = np.argsort(eig_vals)
    rev_sorted_args = args[::-1]
    rev_sorted_vals = eig_vals[rev_sorted_args]
    assert method in {0, 1, 2}, "method parameter must be one of 0,1,2"
    if method == 0:
        print("""Ignoring background noise and using
raw sample covariance estimates...""")
        return rev_sorted_vals
    elif method == 1:
        print("Applying hard thresholding...")
        return np.maximum(rev_sorted_vals,
                          bg_noise_var * np.ones(num_features))
    else:
        print("Applying soft thresholding...")
        tau = comp_sim_tau(rev_sorted_vals, bg_noise_var)
        sim_vars = rev_sorted_vals - tau
        sim_vars[sim_vars < bg_noise_var] = bg_noise_var
        return sim_vars


def comp_sim_tau(rsrtd_vals, bg_noise_var):
    """
    Finds tau to preserve trace of sample cov matrix in the simulation cov matrix:
sum_{i}(lambda_i) = sum_{i}{(lambda_i - tau - bg_noise_var)_{+} + bg_noise_var}
"""
    diffs = rsrtd_vals - bg_noise_var
    expensive_diffs = np.where(diffs <= 0)[0]
    first_nonpos_ind = expensive_diffs[0]
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
    return tau
