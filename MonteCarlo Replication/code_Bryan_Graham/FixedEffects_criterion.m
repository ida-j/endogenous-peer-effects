function [A_i1] = FixedEffects_criterion(A_i0,D,W,beta)

% SUMMARY:          This function is called by the function FixedEffects.m.
%                   It computes one iterate of the fixed point iteration
%                   procedure described in Chatterjee, Diaconis and Sly
%                   (2011, Annals of Applied Probability) and further adapted in Graham (2015).

% INPUTS:           A_i0    : N x 1 vector of starting values of A_i 
%                   D       : N X N adjacency matrix for the network
%                   W       : 0.5*N(N-1) X K matrix of dyad-level covariates 
%                   beta    : K x 1 vector of parameters on W
%                                 

% OUTPUTS:
%                   A_i1    : Next iteration of A_i vector (N x 1)

% CALLED BY                 : FixedEffects()
% FUNCTIONS CALLED          : NA

N = size(D,1);
K = size(W,2);

% Compute r_ij terms which enter the contraction mapping
% Numerator of r_ij
r_ij_1    = exp(W*beta);                      
r_ij_1    = squareform(r_ij_1);               % convert to N X N matrix   

% First term in the denominator of r_ij
A_j     = repmat(A_i0',N,1) - diag(A_i0);     % N x N matrix with A_j in each element in jth column
r_ij_2  = exp(-A_j);                          % N X N matrix

% Second term in the denominator of r_ij
A_i     = repmat(A_i0,1,N) - diag(A_i0);      % N x N matrix with A_i in each element in ith row
r_ij_3  = r_ij_1 .* exp(A_i);                 % N X N matrix

% Calculate r_ij for current value of A_i and beta
r_ij = r_ij_1 ./ (r_ij_2 + r_ij_3);           % N x N matrix (Note diagonal is already zero since numerator diagonal is zero)

% Compute N x 1 vector of average actor-level link probabilities
D_bar   = sum(D,2);                           % N x 1 vector with individual degrees
r_bar   = sum(r_ij,2);                        % N x 1 vector of row sum of r_ij 

% Iterate mapping
A_i1 = log(D_bar) - log(r_bar); 

end

