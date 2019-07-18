function [beta_PL, VCOV_beta_PL, exitflag] = betaSNM_PairwiseLogit(S, C, N, ij_in_tetrad_indicators, silent)

% SUMMARY: This function computes the P=dim(W) vector of common coefficients in the
%          model of link formation studied in Graham (2015) using the pairwise
%          logit (PL) estimator introduced there. To prepare the raw data
%          for use by this function the user will need to use both the
%          GetTetradIndicesForPairwiseLogit() and
%          OrganizeDataForPairwiseLogit() functions.

% INPUTS:   S                       : As returned by OrganizeDataForPairwiseLogit()
%           C                       : As returned by OrganizeDataForPairwiseLogit()
%           N                       : Number of agents
%           ij_in_tetrad_indicators : As returned by GetTetradIndicesForPairwiseLogit()
%           silent                  : if 1 supress optimization output, otherwise do not

% OUTPUTS   beta_PL                 : "pairwise logit" estimate of beta (K x 1 vector)
%           VCOV_beta_PL            : estimate of variance-covariance matrix
%           exitflag                : exitflag as returned by fminunc()


% CALLED BY                 : NA
% FUNCTIONS CALLED          : LOGIT()

%-------------------------------------------------------------------------%
%- STEP 1 : PREPARE DATA FOR ESTIMATION                                  -%
%-------------------------------------------------------------------------%

% Compute size of network and number of regressors
P   = size(C,2);                % number of regressors    
n   = nchoosek(N,2);            % number of unique tetrads 
NC4 = nchoosek(N,4);            % number of unique tetrads    

% drop all quadruples for which S_ijkl = 0 (these cases drop out of the
% objective function)
g   = find(S~=0);
S_t = S(g);
Y_t = 0*(S_t==-1) + 1*(S_t==1);  % convert S into a 0,1 variable
C_t = C(g,:);                    % extract corresponding subset of regressor matrix       

%-------------------------------------------------------------------------%
%- STEP 2 : COMPUTE POINT ESTIMATES FOR BETA                             -%
%-------------------------------------------------------------------------%
[beta_PL, hess_PL, exitflag] = LOGIT(Y_t,C_t,silent);

%-------------------------------------------------------------------------%
%- STEP 3 : COMPUTE VARIANCE-COVARIANCE MATRIX                           -%
%-------------------------------------------------------------------------%

%---------------------------------%
%- Compute covariance of "score" -%
%---------------------------------%

% full "score" vector, including non-contributing tetrads
score_ijkl   = repmat(abs(S) .* (0*(S==-1) + 1*(S==1) - (1 + exp(-C*beta_PL)).^-1),1,P);

% normalize matrices containing all six components of each dyad's
% contribution to (the projection of) the score. These correspond to the 6 non-redundant
% permutations of l_ij,kl which enter the criterion function as described
% in Graham (2015).

score_ij_1 = zeros(n,P);
score_ij_2 = zeros(n,P);
score_ij_3 = zeros(n,P);
score_ij_4 = zeros(n,P);
score_ij_5 = zeros(n,P);
score_ij_6 = zeros(n,P);

% Compute each projection component, regressor-by-regressor
for p=1:P
    score_ijkl_p = score_ijkl .* C(:,p);
    
    % sum of scores associated with tetrads that contain ij (permutation #1);
    score_ij_1(:,p)  = sum(score_ijkl_p(ij_in_tetrad_indicators),2);

    % sum of scores associated with tetrads that contain ij (permutation #2);
    score_ij_2(:,p)  = sum(score_ijkl_p(ij_in_tetrad_indicators+NC4),2);

    % sum of scores associated with tetrads that contain ij (permutation #3);
    score_ij_3(:,p)  = sum(score_ijkl_p(ij_in_tetrad_indicators+2*NC4),2);

    % sum of scores associated with tetrads that contain ij (permutation #4);
    score_ij_4(:,p)  = sum(score_ijkl_p(ij_in_tetrad_indicators+3*NC4),2);

    % sum of scores associated with tetrads that contain ij (permutation #5);
    score_ij_5(:,p)  = sum(score_ijkl_p(ij_in_tetrad_indicators+4*NC4),2);

    % sum of scores associated with tetrads that contain ij (permutation #6);
    score_ij_6(:,p)  = sum(score_ijkl_p(ij_in_tetrad_indicators+5*NC4),2);
end

score_ij    =  (score_ij_1 + score_ij_2 + score_ij_3 + score_ij_4 + score_ij_5 + score_ij_6)/(n - 2*(N-1) + 1);
clear score_iAj_1 score_ij_2 score_ij_3 score_ij_4 score_ij_5 score_ij_6 score_ijkl;

% Compute covariance matrix of score projection; make "degrees-of-freedom"
% adjustment
OMEGA_hat   = cov(score_ij,1)*(n/(n-P));

%----------------------------------%    
%- Compute inverse Hessian matrix -%
%----------------------------------% 

% Compute negative inverse hessian of pairwise logit criterion function
iGAMMA_hat  = inv(-hess_PL/NC4);

% Computer variance-covariance matrix of beta_PL
VCOV_beta_PL = 36*iGAMMA_hat*OMEGA_hat*iGAMMA_hat;