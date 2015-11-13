See /docs for papers giving a formal description of the SigClust algorithm and for the official R documentation on the R function sigclust.

/enwiki_data contains a data set "data2.tsv" of almost 20,000 English Wikipedia article revisions, and the module read.py contains a utility "get_mat" for reading tsv files into numpy arrays to be passed to sigclust.

/tests is limited at this point, but contains some nose tests that runs sigclust on randomly generated data and makes sure the mean p-value returned is not too small or large.

/sigclust contains the module sigclust.py in which are defined not only the main function sigclust and all dependent functions (MAD, cluster_index_2, comp_sim_var, comp_sim_tau) but also a program recclust which recursively applies sigclust to a set of data points and then to all sub-clusters, sub-sub-clusters, etc., until all further clusterings would correspond to a p-value below a certain user-defined cutoff. 
