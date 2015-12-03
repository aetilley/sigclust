import numpy as np

from editquality.feature_lists import enwiki
from sigclust.sigclust import sigclust


def read_data(f, features, rids=False):
    """
    Expect f to have tsv format.

    Reads a set of features and a label from a file one row at a time.
    rids says to expect the first column to be id numbers.
    """
    # Implicitly splits rows on \n
    for line in f:
        # Splits columns on \t
        parts = line.strip().split("\t")

        if rids:
            rev_id = parts[0]
            parts = parts[1:]

        # All but the last column are feature values.
        values = parts[:-1]

        # Last column is a label
        label = parts[-1]

        feature_values = []
        for feature, value in zip(features, values):
            # Each feature knows its type and will perform the right conversion

            if feature.returns == bool:
                # Booleans are weird. bool("False") == True.
                # so you need to string match "True"
                feature_values.append(value == "True")
            else:
                feature_values.append(feature.returns(value))

        row = feature_values[:]
        row.append(label == "True")
        if rids:
            row.insert(0, int(rev_id))

        yield row


def get_mat(file_name, rids=True):
    """
    Read data in file_name into a np. array.

    When rids == False, assumes all columns of the file from file_name are
        feature data except for the last colume which is assumed to be labels.
    When rids==True  assumes in addition that the first column is rev_ids
        and returns a *tuple* of that colum of ids together with the usual
        output np.array
    """

    f = open(file_name)

    rows = list(read_data(f, enwiki.damaging, rids))

    mat = np.array(rows).astype(float)

    # Last column is the label
    labels = mat[:, -1]
    result = mat[:, :-1]

    # If rids then expect first colun to be rev_ids
    if rids:
        rid_col = result[:, 0]
        return rid_col, result[:, 1:], labels
    else:
        return result, labels


def sig_test1(shape, iters=20):
    result = np.zeros(iters)
    for i in np.arange(iters):
        X = np.random.rand(shape[0], shape[1])
        p = sigclust(X, verbose=False)[0]
        result[i] = p
    return result


def RSC(file, rids=True, verbose=True, scale=False):
    rid_col, X = get_mat(file, rids=rids)
    while(True):
        p, clust = sigclust(X, verbose=verbose, scale=scale)
        print("p-value: %f" % p)

        s = sum(clust)
        n_samps = X.shape[0]
        print("The clusters have sizes %d, %d" %
              (n_samps - s, s))
        in0 = input("Remove all points in smallest cluster and re-run "
                    "sigclust? (Enter 'n' to terminate.):")

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

        X = np.delete(X, to_remove, axis=0)
