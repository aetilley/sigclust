# from nose.tools import *
import numpy as np
from sigclust.sigclust import sigclust


def sig_test_1(shape=(50, 50), iters=20):
    """
    Run sigclust on randomly generated datasets and print results.

    :Parameters:
        shape : tuple
            Shape of randomly generated datasets
        iters : int
            Number of iterations
    """
    result = np.zeros(iters)
    for i in np.arange(iters):
        print("Simulating random data of shape %r..." % str(shape))
        X = np.random.rand(shape[0], shape[1])
        print("Running sigclust on generated data...")
        p = sigclust(X)[0]
        result[i] = p
        mu = np.mean(result)
        sig = np.std(result)
    print("The set of %d p-values had\nmean: %f and\n"
          "standard deviation: %f\n" %
          (iters, mu, sig))

    assert(mu < .98,
           "Warning: High average p-value for input matrices of random "
           "normal data.")
    assert(mu > .01,
           "Warning: Low average p-value for input matrices of random "
           "normal data.")



def sig_test_2(shape=(50, 50), iters=20):
    """
    Run sigclust on randomly generated datasets and print results.

    :Parameters:
        shape : tuple
            Shape of randomly generated datasets
        iters : int
            Number of iterations
    """
    result = np.zeros(iters)
    for i in np.arange(iters):
        print("Simulating random data of shape %r..." % str(shape))
        X = np.random.rand(shape[0], shape[1])
        print("Running sigclust on generated data...")
        p = sigclust(X, method = 0)[0]
        result[i] = p
        mu = np.mean(result)
        sig = np.std(result)
    print("The set of %d p-values had\nmean: %f and\n"
          "standard deviation: %f\n" %
          (iters, mu, sig))

    assert(mu < .98,
           "Warning: High average p-value for input matrices of random "
           "normal data.")
    assert(mu > .01,
           "Warning: Low average p-value for input matrices of random "
           "normal data.")
