library(sigclust)

data_2 = read.table("data2.tsv", stringsAsFactors = FALSE)
features_2 = data_2[2:(ncol(data_2)-1)]

# featurs_2 data.frame currently contains character vectors.  Need to replace "True" with 1 and 'False' with 0.

features_2[features_2 == "True"] <- 1
features_2[features_2 == "False"] <- 0

data_mat = data.matrix(features_2)
sig_obj = sigclust(data_mat, nsim = 100, icovest = 1)

print(sig_obj@pval)
