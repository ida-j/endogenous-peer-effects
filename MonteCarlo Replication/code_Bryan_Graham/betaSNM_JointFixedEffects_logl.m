function [cLOGL, cSCORE, cINFO] = betaSNM_JointFixedEffects_logl(beta, A_i_sv, D, W, tol_NFP, MaxIter_NFP, N, K)

% SUMMARY:          This function evaluates the concentrated log-likelihood
%                   of the network formation model described in Graham
%                   (2015). For a given value of the common parameter beta, it
%                   "concentrates out" the A_i (i=1,...,N) incidental
%                   parameters.

% INPUTS:           beta        : K x 1 coefficient vector on W
%                   A_i_sv      : N x 1 vector of starting values for A_i 
%                   D           : N X N adjacency matrix for the network
%                   W           : 0.5N(N-1) x K matrix of dyad-level covariates
%                   tol_NFP     : convergence tolerance for fixed point
%                                 computation of A_i_hat(beta)
%                   MaxIter_NFP : Max # of iterations for fixed point
%                                 computation of A_i_hat(beta)
%                   N           : Number of actors in the network
%                   K           : dim(W)

% OUTPUTS:
%                   cLOGL   : concentrated log-likelihood
%                   cSCORE  : score vector of concentrated log-likelihood
%                   cINFO   : information for concentrated log-likelihood
%                   NoSol   : equals 1 if no fixed point for A_i iteration,
%                             0 otherwise

% CALLED BY             : betaSNM_JointFixedEffects(),
% FUNCTIONS CALLED      : FixedEffects()

%-------------------------------------------------------------------------%
%-Compute MLE of A_i conditional on current value of beta via fixed point-%
% iteration                                                              -%
%-------------------------------------------------------------------------%

% STEP 1: compute A_i(beta) for i=1,...,N
[A_i1, ~, NoSol] = FixedEffects(A_i_sv,D,W,beta,tol_NFP,MaxIter_NFP);

% STEP 2: compute concentrated log-likelihood and first and second
% derivatives with respect to beta

if NoSol==0 % Fixed point iteration successful
    % Form 0.5N(N-1) X 1 vector with A_i + A_j terms
    A_ij = repmat(A_i1,1,N) + repmat(A_i1',N,1) - 2*diag(A_i1);
    A    = squareform(A_ij)';
    
    % Form 0.5N(N-1) X 1 vector with link outcomes
    D    = squareform(D)';

    % Form concentrated likelihood, score and hessian
    exp_Wbeta_A  = exp(W*beta + A);      % Value of index entering link probabilities
    cLOGL        = -(sum(D .* (W*beta + A)) - sum(log(1+exp_Wbeta_A)));
    cSCORE       = -(W'*(D - (exp_Wbeta_A ./ (1+exp_Wbeta_A))));
    cINFO        = ((repmat((exp_Wbeta_A ./ (1+exp_Wbeta_A).^2),1,K) .* W)'*W);
else % Fixed point iteration fails (these functions returns will cause optimization to cease)
    cLOGL   = -9;
    cSCORE  = zeros(K,1);
    cINFO   = eye(K,K);
end

