import numpy as np
from enwiki_data.read import get_mat
from sigclust.sigclust import sigclust

def foo(i):
    return labels[i] == 1

SUB_PRED = foo
"""
Set a  one place predicate (boolean valued)
defining the subpopulation of interest.
"""

ids, features, labels = get_mat(
    "enwiki_data/data2.tsv")

#Sets of row indices for the whole set
# and for the subset of interest.

popul_ind = (np.arange(ids.shape[0]))
subpop_bool = SUB_PRED(popul_ind)
subpop_ind = popul_ind[subpop_bool]

BIG = ids.shape[0]
SMALL = subpop_ind.shape[0]


ids_sub = ids[subpop_bool]
features_sub = features[subpop_bool,:]
labels_sub = labels[subpop_bool]


#First we cluster the whole population
print("Computing p-value for population.")

p, clust = sigclust(features)

print("p-value for population:  %f" % p)
clust0_bool = clust == 0
clust0_ind = popul_ind[clust0_bool]

clust1_bool = ~clust0_bool
clust1_ind = popul_ind[clust1_bool]




#The clusters of the whole population
# determine two subsets
ss_of_clust0_bool = clust0_bool & subpop_bool
ss_of_clust1_bool = clust1_bool & subpop_bool




#Now to cluster the smaller set directly
print("Computing p-value for sub-population.")
p_sub, clust_sub = sigclust(features_sub)
print("p-value for subpopulation:  %f" % p_sub)

subclust0_bool = np.zeros(BIG).astype(bool)
subclust1_bool = np.zeros(BIG).astype(bool)
for j in np.arange(SMALL):
    i = subpop_ind[j]
    if clust_sub[j] == 0:
        subclust0_bool[i] = True
    else:
        subclust1_bool[i] = True
        
subclust0_ind = popul_ind[subclust0_bool]
subclust1_ind = popul_ind[subclust1_bool]


# Now want to test the extent to which 
# subclust{0,1}_ind contained in clust{0,1}_ind
inter_0_0 = subclust0_bool & ss_of_clust0_bool
inter_0_1 = subclust0_bool & ss_of_clust1_bool
inter_1_0 = subclust1_bool & ss_of_clust0_bool
inter_1_1 = subclust1_bool & ss_of_clust1_bool


print("Clustering and then taking the \
respective intersections of the clusters \
with the subset in question gives a \
partition of size (%d, %d)." %
(ss_of_clust0_bool.astype(int).sum(),
 ss_of_clust1_bool.astype(int).sum()))

print("On the other hand, \
clustering the subset in question \
directly gives a partition of size (%d, %d)." %
      (subclust0_bool.astype(int).sum(),
      subclust1_bool.astype(int).sum()))

print("""
These two partitions determine 
a four member partition refinement. 
Specifically, letting 
S = the defined subset of indices
A = The intersection of S with
the first outer cluster
B = The intersection of S with
the second outer cluster
C = First (inner) cluster of S
D = Second (inner) cluster of S
The four possible intersections have sizes 
 |A&C|: %d, |A&D|: %d,
 |B&C|: %d, |B&D|: %d.""" %
      (inter_0_0.astype(int).sum(),      
      inter_0_1.astype(int).sum(),
      inter_1_0.astype(int).sum(),
       inter_1_1.astype(int).sum()))
