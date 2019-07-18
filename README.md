# endogenous-peer-effects
Replication code for the Monte Carlo results in the paper "Estimation of Peer Effects in Endogenous Social Networks - Control Function Approach" by Ida Johnsson and Hyungsik Roger Moon

The contents of this MonteCarlo Replication folder are as follows.

1. functions - functions called in the simulations
2. code_Bryan_Graham - code written by Bryan Graham used to
simulate  network formation and estimate a_i. The code is taken
from replication files of a working paper version of
"An econometric model of network formation with degree heterogeneity" by Bryan Graham


3. Monte Carlo designs - the main folder that contains all code needed
to replicate the results in the paper. This folder contains the following files

- MC_dense_network.m - replicates the MC results for dense network designs using a Hermite polynomial sieve
- MC_dense_network_pol.m - replicates the MC results for dense network designs using a polynomial sieve
- MC_sparse_network.m - replicates the MC results for sparse network designs using a Hermite polynomial sieve
- MC_sparse_network_pol.m - replicates the MC results for sparse network designs using a polynomial sieve
- Dense_network_statistics.m - code that computes and generates tables with summary statistics for the dense network designs
- Sparse_network_statistics.m - code that computes and generates tables with summary statistics for the sparse network designs
- cross_validation_dense.m - code that performs cross-validation for the dense network formation designs using a Hermite polynomial sieve. Can easily be modified to use other sieve functions

To run the files change the cd and path in accordance with your file structure. All .m files generate latex tables and save them in the cd. The tables can be included in a latex file using
\\input\{path_to_table/table_name.tex\} etc.\

