function [beta_ML, INFO, exitflag] = LOGIT(D,X,silent);

% SUMMARY: This function is a standard implementation of LOGIT ML.

% INPUTS
% D                     : N x 1 vector of binary outcomes
% X                     : X is a matrix of covariates (without a constant)
% silent                : when silent = 1 optimization output is suppressed and looser
%                         convergence criteria are used

% OUTPUTS
% phi_ML                : estimates of logit coefficients
% INFO                  : negative Hessian matrix
% exitflag              : exitflag indicator from fminunc()

% CALLED BY             : NA
% FUNCTIONS CALLED      : LOGIT_LOGL

[N K] = size(X);        % Number of observations and covariates
f_logit_logl = @(x)LOGIT_LOGL(x, D, X, K);  % define objective function

% Set optimization parameters
if silent == 1
    % Use Fisher-Scoring with lower tolerances
    options_beta = optimset('LargeScale','on','GradObj','on','Hessian','on',...
                           'Display','off','TolFun',1e-6,'TolX',1e-6,'MaxFunEvals',1000,'MaxIter',1000,'DerivativeCheck','off');
    beta_SV = (X'*X) \ (X'*D);                                 % Linear probability starting values for logit          
else
    % Take a few fisher-scoring steps to get starting values, then do a
    % quasi-newton search with high precision
    options_beta = optimset('LargeScale','on','GradObj','on','Hessian','on',...
                           'Display','iter','TolFun',1e-6,'TolX',1e-6,'MaxFunEvals',10,'MaxIter',10,'DerivativeCheck','off'); 
    
    beta_SV = fminunc(f_logit_logl,  zeros(K,1), options_beta); 
    
    options_beta = optimset('LargeScale','off','GradObj','on','Hessian','off',...
                           'Display','iter','TolFun',1e-12,'TolX',1e-12,'MaxFunEvals',10000,'MaxIter',10000,'DerivativeCheck','off'); 
    
end                                      

[beta_ML,LOGL,exitflag,output,SCORE,INFO] = fminunc(f_logit_logl, beta_SV, options_beta);  % estimate beta_ML    