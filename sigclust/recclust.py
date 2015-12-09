import numpy as np
from .sigclust import sigclust

def recclust(X, threshold=.01,
             mc_iters=100, verbose=True,
             prefix="/", IDS=np.arange(0)):
    """
    Recursively apply sigclust to subclusters of X until all remaining
        clusters have sigclust p-value below threshold.

    Returns dictionary with entries having keys "prefix", "pval", "subclust0",
        "subclust1", and "ids".

    "prefix" is a string representation for a path from the root cluster
        consisting of all samples in the input X.
    "pval" is the p-value of cluster 'prefix'
    "subclust{0,1}" are dictionaries of the same form as this,
        representing the two primary subclusters of the cluster 'prefix'.
        Note that these may be None if pval greater than threshold.
    "ids" is a numpy array of keys to use to record the cluster elements
        themselves.
        Note that this may be None if pval is greater than threshold.
    "tot" is the total number of samples in the cluster.
    """
    if IDS.shape[0] == 0:
        IDS = np.arange(X.shape[0])
    assert IDS.shape[0] == X.shape[0], """Input data and tag list must have
    compatible dimensions (or tag list must be None)."""

    data = {
        "prefix": prefix,
        "pval": None,
        "subclust0": None,
        "subclust1": None,
        "ids": None,
        "tot": 1}

    if X.shape[0] == 1:
        data["ids"] = IDS
        print("Cluster %s has exactly one element." % prefix)
    else:
        p, clust = sigclust(
            X, mc_iters=mc_iters, verbose=verbose)
        print("The p value for\n"
              "subcluster id %s is %f" % (prefix, p))
        data["pval"] = p

        if p >= threshold:
            data["ids"] = IDS
        else:
            pref0 = prefix + "0"
            pref1 = prefix + "1"
            print("Examining sub-clusters\n"
                  "%s and %s" % (pref0, pref1))
            data_0 = X[clust == 0, :]
            data_1 = X[clust == 1, :]
            print("Computing RecClust data\n"
                  "for first cluster.\nPlease wait...")
            dict0 = recclust(data_0,
                             prefix=prefix + "0",
                             IDS=IDS[clust == 0])
            print("Computing Recclust data\n"
                  "for second cluster.\nPlease wait...")
            dict1 = recclust(data_1,
                             prefix=prefix + "1",
                             IDS=IDS[clust == 1])
            data["subclust0"] = dict0
            data["subclust1"] = dict1
            data["tot"] = dict0["tot"] + dict1["tot"]

    return data
