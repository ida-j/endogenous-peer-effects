function [S, C, FracTetrads] = OrganizeDataForPairwiseLogit(D, W, dyad_pair_indices)


% INPUTS:   D                   :   is an N x N undirected adjacency matrix
%           W                   :   is an N x N x K array of dyad-specific covariates (K per
%                                   dyad). Here Z(i,j,:) gives the K-vector of
%                                   dyad-specific covariates for the dyad composed of
%                                   agents i and j.
%           dyad_pair_indices   :   This argumented is outputed by the
%                                   function GetTetradIndicesForPairwiseLogit().

% OUTPUTS:  S                   :   N choose 4 S_ijkl tetrad configuration indicators
%           C                   :  (N choose 4 x K) corresponding matrix of
%                                   covariates.
%           FracTetrads         :   Returns the fraction of all N choose 4
%                                   tetrads that contribute to the pairwise logit criterion

% Compute size of network and dimension of regressor matrix
N    = size(D,1);               % number of agents
P    = size(W,3);               % number of regressors    
NC4  = nchoosek(N,4);           % number of unique tetrads

%-------------------------------------------------------------------------%
%- Construct "outcome" vectors and "regressor" matrices needed to compute-%
%- the variance-covariance matrix for the pairwise logit procedure       -%
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
%- Construct "outcome" vectors and "regressor" matrices needed to compute-%
%- the pairwise-logit point estimate                                     -%
%-------------------------------------------------------------------------%

ij = dyad_pair_indices(:,1);
ik = dyad_pair_indices(:,2);
il = dyad_pair_indices(:,3);
jk = dyad_pair_indices(:,4);
jl = dyad_pair_indices(:,5);
kl = dyad_pair_indices(:,6);

S1a =       D(ij) .*     D(kl) .* (1-D(ik)) .* (1-D(jl))...
      - (1-D(ij)) .* (1-D(kl)) .* D(ik)     .* D(jl);
S1b =       D(ij) .*     D(kl) .* (1-D(il)) .* (1-D(jk))...
      - (1-D(ij)) .* (1-D(kl)) .* D(il)     .* D(jk);
S2a =       D(ik) .*     D(jl) .* (1-D(ij)) .* (1-D(kl))...
      - (1-D(ik)) .* (1-D(jl)) .* D(ij)     .* D(kl);
S2b =       D(ik) .*      D(jl).* (1-D(il)) .*(1-D(jk))...
      - (1-D(ik)) .* (1-D(jl)) .* D(il)     .* D(jk);
S3a =       D(il) .*      D(jk).* (1-D(ij)) .* (1-D(kl))...
      - (1-D(il)) .* (1-D(jk)) .* D(ij)     .* D(kl);
S3b =       D(il) .*     D(jk) .* (1-D(ik)) .* (1-D(jl))...
      - (1-D(il)) .* (1-D(jk)) .* D(ik)     .* D(jl); 


FracTetrads = mean(((abs(S1a)+abs(S1b)+abs(S2a)+abs(S2b)+abs(S3a)+abs(S3b))~=0));
  
C1a = zeros(NC4,P);  
C1b = zeros(NC4,P);  

C2a = zeros(NC4,P);
C2b = zeros(NC4,P);  

C3a = zeros(NC4,P);  
C3b = zeros(NC4,P);  

for p = 1:P
    W_p = W(:,:,p);
    
    C1a(:,p) = W_p(ij) + W_p(kl) - ( W_p(ik) + W_p(jl) ); 
    C1b(:,p) = W_p(ij) + W_p(kl) - ( W_p(il) + W_p(jk) ); 
    
    C2a(:,p) = W_p(ik) + W_p(jl) - ( W_p(ij) + W_p(kl) ); 
    C2b(:,p) = W_p(ik) + W_p(jl) - ( W_p(il) + W_p(jk) ); 
    
    C3a(:,p) = W_p(il) + W_p(jk) - ( W_p(ij) + W_p(kl) ); 
    C3b(:,p) = W_p(il) + W_p(jk) - ( W_p(ik) + W_p(jl) ); 
end

S = [S1a; S1b; S2a; S2b; S3a; S3b];
C = [C1a; C1b; C2a; C2b; C3a; C3b];

end

