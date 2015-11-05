from sys import argv

def read_data(f, features, rids=False):
    """
    Reads a set of features and a label from a file one row at a time.
    rids says to expect the first column to be id numbers.
    """
    for line in f: # Implicitly splits rows on \n
        parts = line.strip().split("\t") # Splits columns on \t

        if rids:
            rev_id = parts[0]
            parts = parts[1:]
            
        values = parts[:-1] # All but the last column are feature values.
        label = parts[-1] # Last column is a label

        feature_values = []
        for feature, value in zip(features, values): 
            # Each feature knows its type and will perform the right conversion
            
            if feature.returns == bool:
                # Booleans are weird.  bool("False") == True, so you need to string match "True"
                feature_values.append(value == "True")
            else:
                feature_values.append(feature.returns(value))

        row = feature_values[:]
        row.append(label == "True")
        if rids:
            row.insert(0, int(rev_id))
                
        yield row



from editquality.feature_lists import enwiki
import numpy as np
from sigclust.sigclust import sigclust

def get_mat(file_name, rids = False):
    """
    Reads data in file_name into a np. array.
    When rids == False, assumes all columns of the file from file_name are feature data except for the last colume which is assumed to be labels.
    When rids==True  assumes in addition that the first column is rev_ids and returns a *tuple* of that colum of ids together with the usual output np.array
    """

    f = open(file_name)

    rows = list(read_data(f, enwiki.damaging, rids))

    mat = np.array(rows).astype(float)

    #Last column is the label
    labels = mat[:, -1]
    result = mat[:, :-1]

    #if rids then expect first colun to be rev_ids
    if rids:
        rid_col = result[:, 0]
        return rid_col, result[:, 1:], labels
    else:
        return result, labels


"""

 Note:  currently data_mat is about half zeros.

When running sigclust 30 times on normally generated data of size (20, 5) with mc_iters = 1000 and floor = 0,  the resulting p values were

Out[6]: 
array([ 0.218,  0.367,  0.656,  0.34 ,  0.014,  0.208,  0.526,  0.791,
        0.662,  0.645,  0.128,  0.607,  0.23 ,  0.796,  0.449,  0.889,
        0.68 ,  0.499,  0.233,  0.004])

In [7]:
"""
def sig_test1(shape, iters = 20):
    result = np.zeros(iters)
    for i in np.arange(iters):
        X = np.random.rand(shape[0], shape[1])
        p = sigclust(X, verbose = False)[0]
        result[i] = p
    return result
        

"""Note:  
Running sig_test((20, 5), 30) gives

array([ 0.79 ,  0.705,  0.155,  0.53 ,  0.08 ,  0.235,  0.625,  0.03 ,
        0.555,  0.765,  0.2  ,  0.23 ,  0.545,  0.685,  0.635,  0.815,
        0.795,  0.815,  0.575,  0.07 ,  0.685,  0.3  ,  0.875,  0.385,
        0.74 ,  0.955,  0.395,  0.195,  0.13 ,  0.69 ])

Running sig_test(data.shape, 30) (where data is the get_mat("enwiki_data/data1.tsv"))

runs too slowly, but the ci in the first iteration is .983370 and the simulated cis are all around .9847 suggesting a first p-value of 0.

"""

def RSC(file, rids=True, verbose = True, scale = False):
    rid_col, X = get_mat(file, rids = rids)
    while(True):
        p, clust = sigclust(X, verbose = verbose, scale = scale)
        print("p-value: %f" % p)

        s = sum(clust)
        n_samps = X.shape[0]
        print("The clusters have sizes %d, %d" %
              (n_samps - s, s))
        in0 = input("Remove all points in smallest cluster and re-run sigclust?  (Enter 'n' to terminate.):")
        
        if in0 is 'n':
            break

        
        sec_small = s < (n_samps / 2)
        print("Removing %s cluster (of size %d)." %
              ("SECOND" if sec_small else "FIRST",
               s if sec_small else n_samps - s))
        
               

        f_clust = clust.astype(bool)
        if sec_small:
            to_remove = np.where(f_clust)[0]
        else:
            to_remove = np.where(~f_clust)[0]
        print("Now removing samples with the following indices:")
        print(to_remove)
        print("These samples correspond to the following rev ids:")
        rem_rids = rid_col[to_remove]
        
        print(rem_rids)
    
        X = np.delete(X, to_remove, axis = 0)
