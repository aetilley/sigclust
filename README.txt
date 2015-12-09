This package contains a Python implimentation of the SigClust algorithm originally proposed in [1] and whose soft-thresholding variant is given in [2].

See /docs for papers giving a formal description of the SigClust algorithm and for the official R documentation on the R package sigclust.

/sigclust contains the module sigclust.py in which the sigclust function is defined as are the functions MAD, cluster_index_2, comp_sim_var, comp_sim_tau used by the sigclust function.
/sigclust also contains a module recclust.py containing a function recclust, which recursively applies sigclust to a data matrix and then to all sub-clusters, sub-sub-clusters, etc., until all further clusterings would correspond to a p-value above a certain user-defined cutoff.

/enwiki_data contains data2.tsv whose near 20,000 rows each represent feature values for English Wikipedia article revisions and whose last column gives target labels of "damaging / non-damaging."
/enwiki_data also contains a module read.py containing a utility "get_mat" for reading tsv files into numpy arrays to be passed to sigclust.

/tests is limited at this point, but contains some nose tests that runs sigclust on randomly generated data and makes sure the mean p-value returned is not too small or large.  There is also an R script R_read.R which runs the R sigclust function on the enwiki_data and some randomly generated data.

This package was developed by Arthur Tilley and sprung out of interest in clustering Wikipedia article revisions while funded by an Individual Engagement Grant from the Wikimedia Foundation.

Arthur can be reached at aetilley at gmail.

[1]  Yufeng Liu, David Neil Hayes, Andrew Nobel and J. S. Marron, Statistical Significance of Clustering for High-Dimension, Low-Sample Size Data, Journal of the American Statistical Association, Vol. 103, No. 483 (Sep., 2008), pp. 1281-1293

[2] Hanwen Huang, Yufeng Liu, Ming Yuan, J.S. Marron, Statistical Significance of Clustering using Soft Thresholding, Journal of Computational and Graphical Statistics, 2014 (DOI:10.1080/10618600.2014.948179)
