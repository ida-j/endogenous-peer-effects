function [bias_hat_jfe, VCOV_hat_jfe, A_iter] = betaSNM_JointFixedEffects_BV(beta, A, D, W, T, tol_NFP, MaxIter_NFP, iterate, obs_hs)

% SUMMARY:          This function computes asymptotic bias and variance estimates 
%                   for the common parameter in the network formation model
%                   studied by Graham (2015).

% INPUTS:           beta    : K x 1 coefficient vector on W
%                   A       : N x 1 vector of values for A_i 
%                   D       : N X N adjacency matrix for the network
%                   W       : 0.5N(N-1) x K matrix of dyad-level covariates
%                   T       : 0.5N(N-1) x N selection matrix for A
%                   tol_NFP : Convergence criterion for fixed point iteration computation of
%                             A_i(beta) and also for iterated bias
%                             correction convergence
%             MaxIter_NFP   : Max number of fixed point iterations computation of
%                             A_i(beta)
%                   iterate : if iterate = 1 perform iterated bias correction,
%                             single step correction otherwise
%                   obs_hs  : if obs_hs = 1 use the observe H_AA Hessian in place of
%                             V_N for Bias correction.

% OUTPUTS:
%                   bias_hat_jfe   : asymptotic bias estimate
%                   VCOV_hat_jfe   : asymptotic variance estimate
%                   A_iter         : Estimates of A at bias corrected value
%                                    for beta

% CALLED BY             : betaSNM_JointFixedEffects()
% FUNCTIONS CALLED      : LOGIT_LOGL()


% find values for N, n=0.5N(N-1) and K
N       = size(D,2);
[n K]   = size(W);

% initialize parameters for iterated bias correction while loop
done        = 0;
beta_iter   = beta;
A_iter      = A;

% compute 0.5N(N-1) x 1 vectorization of adjacency matrix & degree sequence
% vector
D_ij    = squareform(D)';
Ds      = sum(D,2);

while (done == 0)
        
    % get hessian at current value of beta
    [f_LOGL, f_SCORE, f_OBS_INFO] = LOGIT_LOGL([beta_iter; A_iter],D_ij,[W T],K+N);
    
    % Compute 0.5N(N-1) vector of fitted link probabilities
    p_ij = (1+exp(-[W T]*[beta_iter; A_iter])).^-1;
    
    % Calculate hessian of concentrated likelihood
    H_bb        = -f_OBS_INFO(1:K,1:K);
    H_bA        = -f_OBS_INFO(1:K,(K+1):(K+N));
    H_AA        = -f_OBS_INFO((K+1):(K+N),(K+1):(K+N));
        
    % Calculate matrices used for Hessian approximation
    p_ij        = squareform(p_ij);                     % N x N "fitted/expected" adjacency matrix
    iV_N        = diag(sum(p_ij .* (1 - p_ij)).^-1);    % inverse of V_N matrix
    Q           = iV_N - sum(sum(p_ij .* (1 - p_ij))).^-1*ones(N,N); % Q matrix
    
    if (obs_hs == 1)
        % Compute Fisher Information for beta
        INFO_hat        = -(H_bb - H_bA*inv(H_AA)*H_bA')/n;
        iINFO_hat       = inv(INFO_hat);
        VCOV_hat_jfe    = iINFO_hat;
    else
        % Approximate Fisher Information
        INFO_hat        = -(H_bb + H_bA*Q*H_bA')/n;
        iINFO_hat       = inv(INFO_hat);
        VCOV_hat_jfe    = iINFO_hat;
    end
    
    % Calculate K-vector of bias terms
    s_b_AA = -p_ij .* (1 - p_ij) .* (1 - 2*p_ij);
    s_A   = Ds - sum(p_ij,2);   % heterogeneity score sub-vector 
    
    % Build of bias expression element-by-element
    bias_hat_jfe = zeros(K,1);
    for k = 1:K
        s_b_AA_k            = s_b_AA .* squareform(W(:,k));
        bias_hat_jfe(k,1)   = 0.5*(sum(sum(s_b_AA_k) .* diag(iV_N)'));
    end 

        
    % Bias estimate for common parameter   
    bias_hat_jfe = iINFO_hat*bias_hat_jfe/n;
    
    if (iterate == 1)
        % If iterate = check for convergence
        if (norm(beta_iter - (beta - bias_hat_jfe)) <= tol_NFP);
            done = 1;                             % If convergence, then terminate loop
        else
            beta_iter   = beta - bias_hat_jfe;    % Otherwise update beta_iter
            A_iter      = FixedEffects(A_iter,D,W,beta_iter,tol_NFP, MaxIter_NFP);
        end
    else
        % If iterate = 0 terminate loop after single run
        done = 1;        
    end
end