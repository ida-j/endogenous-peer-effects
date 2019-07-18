function [LOGL, SCORE, INFO] = LOGIT_LOGL(beta,D,X,K)

% SUMMARY: This function evaluates the logit log-likelihood as well
%          as its first and second derivatives with respect to the
%          coefficient vector.

% INPUTS
% beta         : covariate vector at which log-likelihood is evaluated
% D            : N x 1 vector of binary outcomes
% X            : X is a matrix of covariates (without a constant)
% K            : dimension of beta

% OUTPUTS
% LOGL         : negative of log-likelihood
% SCORE        : first derivative vector
% INFO         : second derivative matrix

% CALLED BY             : LOGIT()
% FUNCTIONS CALLED      : NA

exp_Xbeta   = exp(X*beta);
LOGL        = -(sum(D .* (X*beta)) - sum(log(1+exp_Xbeta)));
SCORE       = -(X'*(D - (exp_Xbeta ./ (1+exp_Xbeta))));
INFO        = ((repmat((exp_Xbeta ./ (1+exp_Xbeta).^2),1,K) .* X)'*X);