function [beta_hat_jfe, bias_hat_jfe, A_i_hat_jfe, VCOV_hat_jfe, exitflag, NumFPIter] = betaSNM_JointFixedEffects(beta_sv, A_i_sv, D, W, T, tol_NFP, MaxIter_NFP, silent, iterate, obs_hs);

% SUMMARY: This function computes coefficient estimates in the extended
%          beta model of social network formation by the method of *joint*
%          maximum likelihood. The model and estimation algorithmn is
%          an extension of the one discussed in Chatterjee,
%          Diaconis and Sly (2011, Annals of Applied Probability) to
%          include dyad level covariates as described in Graham (2015).

% INPUTS

% beta_sv       : K x 1 vector of starting values for beta
% A_i_sv        : N x 1 vector of starting values for A_i heterogeneity terms
% D             : N x N adjacency matrix 
% W             : 0.5*N(N-1) x K matrix of dyad-level covariates
% T             : 0.5*N(N-1) x N selection matrix for A vector
% tol_NFP       : Convergence criterion for fixed point iteration computation of
%                 A_i(beta)
% MaxIter_NFP   : Max number of fixed point iterations computation of
%                 A_i(beta)
% silent        : when silent = 1 optimization output is suppressed and
%                 optimization is by Fisher-scoring with lower tolerances.
%                 Otherwise optimization starts with a few Fisher-scoring steps
%                 and then switches to a quasi-Newton search.
% iterate       : if iterate=1, then perform iterated bias correction,
%                 perform single step correction otherwise
% obs_hs        : if 1 use the observe H_AA Hessian in place of
%                 V_N for Bias correction.

% OUTPUTS

% beta_hat_jfe          : joint FE MLE estimates of beta coefficients
% bias_hat_jfe          : asymptotic bias estimate (divided by sqrt(n))
% A_i_hat_jfe           : joint FE MLE estimates of A_i terms 
% VCOV_beta_hat_jfe     : large sample covariance of beta_hat (= negative inverse Hessian)
% exitflag              : exitflag as return by fminunc(), set = -9 in
%                         concentrated MLEs of the A_i (i=1,...,N) do not exist.
% NumFPIter             : number of fixed point iterations to compute final
%                         estimate of A_i

% CALLED BY             : NA
% FUNCTIONS CALLED      : betaSNM_JointFixedEffect_logl()
%                       : betaSNM_JointFixedEffects_BV()
%                       : FixedEffects()

% Compute size of network and dimension of regressor matrix
N = size(D,1);
n = 0.5*N*(N-1);
K = size(W,2);

% Define concentrated log likelihood criterion function
f_betaSNM_JointFixedEffect_logl = @(x)betaSNM_JointFixedEffects_logl(x, A_i_sv, D, W, tol_NFP, MaxIter_NFP, N, K);  % define objective function

% Set optimization parameters
if silent == 1      
    % Use Fisher-Scoring with lower tolerances
    options_beta = optimset('LargeScale','on','GradObj','on','Hessian','on',...
                             'Display','off','TolFun',1e-6,'TolX',1e-6,'MaxFunEvals',100,'MaxIter',25);      
else
    % Take a few Fisher-Scoring steps to get starting values, then do a
    % quasi-newton search; use higher tolerance
    options_beta = optimset('LargeScale','on','GradObj','on','Hessian','on',...
                             'Display','iter','TolFun',1e-6,'TolX',1e-6,'MaxFunEvals',10,'MaxIter',10); 
    
    beta_sv = fminunc(f_betaSNM_JointFixedEffect_logl, beta_sv, options_beta); 
    
    options_beta = optimset('LargeScale','on','GradObj','on','Hessian','on',...
                             'Display','iter','TolFun',1e-20,'TolX',1e-12,'MaxFunEvals',1000,'MaxIter',1000); 
    
end                                      

%-------------------------------------------------------------------------%
%- STEP 1 : COMPUTE JOINT MLE                                            -%
%-------------------------------------------------------------------------%

% Calculate joint MLE of beta by alternating between fixed point iteration
% to solve for A_i(beta) and a concentrated MLE update to find beta_hat as
% detailed in Graham (2015). This is a "see-saw" method.
[beta_hat_jfe,cLOGL,exitflag,output,cSCORE,cINFO] = fminunc(f_betaSNM_JointFixedEffect_logl, beta_sv, options_beta);  % estimate beta_hat_jfe    

% Solve for MLEs of the A_i vector by fixed point iteration
[A_i_hat_jfe, NumFPIter] = FixedEffects(A_i_sv, D, W, beta_hat_jfe, tol_NFP, MaxIter_NFP);

%-------------------------------------------------------------------------%
%- STEP 2 : COMPUTE VARIANCE-COVARIANCE MATRIX & BIAS ESTIMATE           -%
%-------------------------------------------------------------------------%
if sum(isnan(A_i_hat_jfe))>0
    % If MLE does not exist, then set exitflag to -9 and other function
    % output to NaN
    exitflag = -9;
    bias_hat_jfe = NaN*ones(K,1);
    VCOV_hat_jfe = NaN*ones(K,K);
    disp('WARNING: No fixed point, MLE does not exist');
else
    % If MLE does exist, use betaSNM_JointFixedEffects_BV() function to get
    % estimates of the asymptotic bias and covariance matrix
    [bias_hat_jfe, VCOV_hat_jfe, A_i_hat_jfe] = betaSNM_JointFixedEffects_BV(beta_hat_jfe, A_i_hat_jfe, D, W, T, tol_NFP, MaxIter_NFP, iterate, obs_hs);
end




