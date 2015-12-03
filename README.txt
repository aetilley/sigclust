Implimentation of the SigClust algorithm originally proposed in [1] and whose soft-thresholding variant is given in [2].

See /docs for papers giving a formal description of the SigClust algorithm and for the official R documentation on the R function sigclust.

/enwiki_data contains data2.tsv whose near 20,000 rows each represent feature values for English Wikipedia article revisions, and the module read.py contains a utility "get_mat" for reading tsv files into numpy arrays to be passed to sigclust.

/tests is limited at this point, but contains some nose tests that runs sigclust on randomly generated data and makes sure the mean p-value returned is not too small or large.  There is also an R script R_read.R which runs the R sigclust function on the enwiki_data and some randomly generated data.

/sigclust contains the module sigclust.py in which are defined not only the main function sigclust and all dependent functions (MAD, cluster_index_2, comp_sim_var, comp_sim_tau) but also a program recclust which recursively applies sigclust to a set of data points and then to all sub-clusters, sub-sub-clusters, etc., until all further clusterings would correspond to a p-value below a certain user-defined cutoff. 

The original version of this code sprung out of interest in clustering Wikipedia article revisions while funded by an Individual Engagement Grant from the Wikimedia Foundation.  

[1]  Yufeng Liu, David Neil Hayes, Andrew Nobel and J. S. Marron, Statistical Significance of Clustering for High-Dimension, Low-Sample Size Data, Journal of the American Statistical Association, Vol. 103, No. 483 (Sep., 2008), pp. 1281-1293

[2] Hanwen Huang, Yufeng Liu, Ming Yuan, J.S. Marron, Statistical Significance of Clustering using Soft Thresholding, Journal of Computational and Graphical Statistics, 2014 (DOI:10.1080/10618600.2014.948179)
