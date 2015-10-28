library(sigclust)

data_2 = read.table("data2.tsv", stringsAsFactors = FALSE)
features_2 = data_2[2:(ncol(data_2)-1)]

# featurs_2 data.frame currently contains character vectors.  Need to replace "True" with 1 and 'False' with 0.

features_2[features_2 == "True"] <- 1
features_2[features_2 == "False"] <- 0

data_mat = data.matrix(features_2)
sig_obj = sigclust(data_mat, nsim = 100, icovest = 3)

print(sprintf("pval for input data:  %f", sig_obj@pval))

##TEST on random data of same dim.
nrow = dim(data_mat)[1]
ncol = dim(data_mat)[2]
tot_vals = length(data_mat)

rand_mat_0 = matrix(rnorm(tot_vals, mean = 0, sd = 1), nrow = nrow, ncol=ncol)

sig_obj_0 = sigclust(rand_mat_0, nsim = 100, icovest = 3)
print(sprintf("pval for random matrix of same size:  %f", sig_obj_0@pval))

## Two halves from their own normal
ncol_1_1 = floor(ncol / 2)
ncol_1_2 = ncol - ncol_1_1

rand_mat_1_1 = matrix(rnorm(nrow*ncol_1_1, mean = 0, sd = 1), nrow = nrow, ncol = ncol_1_1)


rand_mat_1_2 = matrix(rnorm(nrow*ncol_1_2, mean = .5, sd = 1), nrow = nrow, ncol = ncol_1_2)

rand_mat_1 = rbind(rand_mat_1_1, rand_mat_1_2)

sig_obj_1 = sigclust(rand_mat_1, nsim= 100, icovest = 3)
print(sprintf("Letting half of input data be from N(0,1) and hald be from N(.5, 1):  %f", sig_obj_1@pval))