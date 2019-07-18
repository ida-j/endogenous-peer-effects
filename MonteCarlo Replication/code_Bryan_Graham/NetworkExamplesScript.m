%---------------------------------------------------------%
%- This script reproduces the Monte Carlo results        -%
%- in Graham (2015)                                      -%
%- by Bryan S. Graham, UC - Berkeley                     -%
%---------------------------------------------------------%

clear all;
N = 100;                                    % Number of agents

addpath(genpath('~/Dropbox/Ida (1)/Draft/MonteCarlo/code_Bryan/'))
% Add directory to path and change current directory
%path(path,'/accounts/fac/bgraham/Research_EML/DynamicNetworks/Empirics/Matlab_Code');
%cd('/accounts/fac/bgraham/Research_EML/DynamicNetworks/Empirics/Matlab_Code');

% delete(['ExogNetLogFile_N_' num2str(N) '.log']);
% diary(['ExogNetLogFile_N_' num2str(N) '.log']);
% diary off;A

% Open up a parallel pool
ParallelPool = parpool;

%---------------------------------------------------------------%
%- Construct incidence matrices for sparse matrix calculations -%
%---------------------------------------------------------------%

% Calculate the required tetrad indices for pairwise logit estimation
%[dyad_pair_indices, ij_in_tetrad_indicators] = GetTetradIndicesForPairwiseLogit(N);
%save(['mc_network_indices_N_' num2str(N) '.mat'], 'dyad_pair_indices', 'ij_in_tetrad_indicators');
%clear dyad_pair_indices ij_in_tetrad_indicators;

% Load stored dyad and tetrad indice matrices for sparse matrix calculation
% for current sample size
sparse_calc_indices = ['mc_network_indices_N_' num2str(N)]; 
load(sparse_calc_indices);

%---------------------------------------------------------%
%- Set up Monte Carlo Data Generating Process # 1        -%
%---------------------------------------------------------%

% NOTES: "two-type" model
n = 0.5*N*(N-1);                           % Number of dyads     
K = 1;                                     % Number of dyadic regressors 
B = 10;                                  % Number of MC replications
 
% Optimmization parameters
beta_sv     = zeros(K,1);                  % Starting value for beta_sv 
A_i_sv      = zeros(N,1);                  % Starting values for A_i vector 
tol_NFP     = 1e-6;                        % Convergence criterion for fixed point iteration step 
MaxIter_NFP = 100;                         % Maximum number of NFP iteractions 
silent      = 1;                           % Show optimization output (or not) 
iterate     = 1;                           % Iterated bias correction     
obs_hs      = 1;                           % Used observed H_AA hessian instead of approximation for bias and variance estimation     

% Compute 0.5N(N-1) x N matrix with T_ij terms
T = zeros(n,N);     % pre-allocate storage space for this matrix
for i = 1:(N-1)
    T(((n-(N-(i-1))*(N-i)/2) + 1):(n-(N-i)*(N-i-1)/2),:) = [zeros(N-i,i-1) ones(N-i,1) eye(N-i)];        
end

% Specify 15 basic designs and initialize Monte Carlo results matrices
% Order of design parameters: frequency of X = 1, alpha0, alpha1, SuppLength, 
% mean of A_i (X=0), mean of A_i (X=1), beta 
Designs = [ 0.5     1       1       1   0            0           1;
            0.5     1       1       1   -0.25       -0.25        1;
            0.5     1       1       1   -0.75       -0.75        1;
            0.5     1       1       1   -1.25       -1.25        1;
            0.5     1/4     3/4     1   0            0.5         1;
            0.5     1/4     3/4     1   -0.5         0           1;
            0.5     1/4     3/4     1   -1          -0.5         1;...
            0.5     1/4     3/4     1   -1.5        -1           1];

if 2==1
        
        
for d = 1:8
        %-------------------------------------------------------------------%
        %- Draw regressor matrix and heterogeneity parameters for design d -%
        %-------------------------------------------------------------------%
        pX          = Designs(d,1);
        AShape1     = Designs(d,2);
        AShape2     = Designs(d,3);
        ASuppLgth   = Designs(d,4);
        AMean0      = Designs(d,5);
        AMean1      = Designs(d,6);
        beta        = Designs(d,7);  
        
        %---------------------------------------------------------%
        %- Run Monte Carlo Experiment                            -%
        %---------------------------------------------------------%
        
        MC_Results_ml       = zeros(B,5*K+1);                                      % Storage matrix for naive ML MC results
        MC_Results_jfe      = zeros(B,8*K+2);                                      % Storage matrix for joint fixed effects MC results
        MC_Results_pl       = zeros(B,5*K+2);                                      % Storage matrix for pairwise logit MC results
        MC_Results_design   = zeros(B,7);                                          % Storage matrix for design features
        
        % Set random number seed
        rng(9);

        parfor b = 1:B
            b
            % Draw observed agent-specific covariate: X = -1 or 1
            X_i    = 2*(random('bino',ones(N,1),pX*ones(N,1))-1/2);     
           
            X_ij   = repmat(X_i,1,N) + repmat(X_i',N,1)  - 2*diag(X_i);
            X      = squareform(X_ij)';
            
            % From W matrix (0.5N(N-1) X K) 
            W_ij   = repmat(X_i,1,N) .* repmat(X_i',N,1) - eye(N);           % N x N matrix with dyad-specific regressor (interaction)
            W      = squareform(W_ij)';                                      % 0.5N(N-1) X 1 vector with dyad-specific regressor
                                                       
            % Draw actor-specific heterogeneity
            A_i = AMean0*(X_i==-1) + AMean1*(X_i==1) ...
                + ASuppLgth*(random('beta',AShape1*ones(N,1),AShape2*ones(N,1)) - AShape1/(AShape1+AShape2));          

            % form 0.5N(N-1) X 1 vector with A_i + A_j terms
            A_ij = repmat(A_i,1,N) + repmat(A_i',N,1) - 2*diag(A_i);
            A    = squareform(A_ij)';

            % 0.5N(N-1) X 1 vector with ij link probabilities
            p    = exp(W*beta + A) ./ (1 + exp(W*beta + A));
    
            % Take random draw from network model for current design
            U = random('unif',zeros(0.5*N*(N-1),1),ones(0.5*N*(N-1),1));    % 0.5N(N-1) X 1 vector of [0,1] uniforms
            D = (U<=p); 
            D_ij = squareform(D);                                           % N x N adjacency matrix
        
            %--------------------------------------------------%
            %- # 1: Compute dyadic-logit estimates of beta    -%
            %--------------------------------------------------%
                       
            [theta_ml, hess_ml, exitflag_ml] = LOGIT(D,[X W],silent);
            beta_hat_ml = theta_ml(K+1:2*K);
            VCOV_hat_ml = inv(hess_ml/n);
            VCOV_hat_ml = VCOV_hat_ml(K+1:2*K,K+1:2*K);
            
            %--------------------------------------------------%
            %- # 2: Compute pairwise logit estimates of beta  -%
            %--------------------------------------------------%
            [S, C, FracTetrads] = OrganizeDataForPairwiseLogit(D_ij, reshape(squareform(W),N,N,K), dyad_pair_indices);
            [beta_hat_pl, VCOV_hat_pl, exitflag_pl] = betaSNM_PairwiseLogit(S, C, N, ij_in_tetrad_indicators, silent);
                        
            %--------------------------------------------------%
            %- # 3: Compute joint MLE estimates of beta and A -%
            %--------------------------------------------------%
            [beta_hat_jfe, bias_hat_jfe, A_i_hat_jfe, VCOV_hat_jfe, exitflag, NumFPIter] = betaSNM_JointFixedEffects(beta_sv, A_i_sv, D_ij, W, T, tol_NFP, MaxIter_NFP, silent, iterate, obs_hs);
                        
            %-----------------------------------------------------%
            % Calculate actual size of t-test for each estimator -%
            %-----------------------------------------------------%
            % alpha = 0.05
            size_05_ml         = 1 - (beta_hat_ml  - 1.96*sqrt(diag(VCOV_hat_ml/n)) <= beta)                        .* (beta_hat_ml + 1.96*sqrt(diag(VCOV_hat_ml/n)) >= beta);
            size_05_pl         = 1 - (beta_hat_pl  - 1.96*sqrt(diag(VCOV_hat_pl/n)) <= beta)                        .* (beta_hat_pl + 1.96*sqrt(diag(VCOV_hat_pl/n)) >= beta);
            size_05_jfe        = 1 - (beta_hat_jfe - 1.96*sqrt(diag(VCOV_hat_jfe/n)) <= beta)                       .* (beta_hat_jfe + 1.96*sqrt(diag(VCOV_hat_jfe/n)) >= beta);
            size_05_jfe_bc     = 1 - (beta_hat_jfe - bias_hat_jfe - 1.96*sqrt(diag(VCOV_hat_jfe/n)) <= beta)        .* (beta_hat_jfe - bias_hat_jfe + 1.96*sqrt(diag(VCOV_hat_jfe/n)) >= beta);
           
            % alpha = 0.10
            size_10_ml         = 1 - (beta_hat_ml  - 1.645*sqrt(diag(VCOV_hat_ml/n)) <= beta)                        .* (beta_hat_ml + 1.645*sqrt(diag(VCOV_hat_ml/n)) >= beta);
            size_10_pl         = 1 - (beta_hat_pl  - 1.645*sqrt(diag(VCOV_hat_pl/n)) <= beta)                       .* (beta_hat_pl + 1.645*sqrt(diag(VCOV_hat_pl/n)) >= beta);
            size_10_jfe        = 1 - (beta_hat_jfe - 1.645*sqrt(diag(VCOV_hat_jfe/n)) <= beta)                      .* (beta_hat_jfe + 1.645*sqrt(diag(VCOV_hat_jfe/n)) >= beta);
            size_10_jfe_bc     = 1 - (beta_hat_jfe - bias_hat_jfe - 1.645*sqrt(diag(VCOV_hat_jfe/n)) <= beta)       .* (beta_hat_jfe - bias_hat_jfe + 1.645*sqrt(diag(VCOV_hat_jfe/n)) >= beta);
            
            % Store bth iteration's MC results
            MC_Results_ml(b,:)      = [beta' beta_hat_ml' sqrt(diag(VCOV_hat_ml)/n)' size_05_ml' size_10_ml' (exitflag_ml>0)];
            MC_Results_pl(b,:)      = [beta' beta_hat_pl' sqrt(diag(VCOV_hat_pl)/n)' size_05_pl' size_10_pl' (exitflag_pl>0) FracTetrads];
            MC_Results_jfe(b,:)     = [beta' beta_hat_jfe' bias_hat_jfe' sqrt(diag(VCOV_hat_jfe/n))' size_05_jfe' size_05_jfe_bc' size_10_jfe' size_10_jfe_bc' (exitflag>0) NumFPIter];                       
            
            DegreeDis = sum(D_ij);
            MC_Results_design(b,:)  = [mean(D) mean(DegreeDis) median(DegreeDis) std(DegreeDis) skewness(DegreeDis,0) min(DegreeDis) max(DegreeDis)];
                        
        end
      
        %---------------------------------------------------------%
        %- Summarize Monte Carlo Results                         -%
        %---------------------------------------------------------%

        % Dyadid Logit / ML Results
        beta0       = MC_Results_ml(:,1:K);
        beta_ML     = MC_Results_ml(:,K+1:2*K);
        
        % Pairwise Logit Results
        beta_PL     = MC_Results_pl(:,K+1:2*K);
        
        % Joint ML Results
        % Calculate bias 
        beta_JML    = MC_Results_jfe(:,K+1:2*K);                                                             % JML point estimate
        beta_BC     = MC_Results_jfe(:,K+1:2*K) - MC_Results_jfe(:,2*K+1:3*K);                               % JML bias-corrected   
         
        % find successul PL and JML estimation replications
        b_ML = find(MC_Results_ml(:,end)==1);
        b_PL = find(MC_Results_pl(:,end-1)==1);
        b_JML = find(MC_Results_jfe(:,end-1)==1);
        
        
        % calculate rmse for each estimator
        rmse_ML     = sqrt(mean((beta_ML(b_ML)      - beta0(b_ML)).^2));
        rmse_PL     = sqrt(mean((beta_PL(b_PL)      - beta0(b_PL)).^2));
        rmse_JML    = sqrt(mean((beta_JML(b_JML)    - beta0(b_JML)).^2));
        rmse_BC     = sqrt(mean((beta_BC(b_JML)     - beta0(b_JML)).^2));
        
        format short;
        diary on;
        disp('');
        disp('_________________________________________________________________________');
        disp('- SUMMARY OF MONTE CARLO RESULTS                                        -');
        disp('_________________________________________________________________________');
        disp(['Number of agents, N      : ' num2str(N)]);
        disp(['Number of dyads, n       : ' num2str(n)]);
        disp(['Number of tetrads,       : ' num2str(nchoosek(N,4))]);
        disp(['Avg. frac of tetrads     : ' num2str(mean(MC_Results_pl(:,end)))]);
        disp(['# of Monte Carlos reps   : ' num2str(B)]);
        disp('_________________________________________________________________________');
        disp('- DATA GENERATING PROCESS                                               -');
        disp('_________________________________________________________________________');
        disp(['Beta                     : ' num2str(Designs(d,7))]);
        disp(['alpha0                   : ' num2str(Designs(d,2))]);
        disp(['alpha1                   : ' num2str(Designs(d,3))]);
        disp(['A_i support length       : ' num2str(Designs(d,4))]);
        disp(['A_i mean | X = -1        : ' num2str(Designs(d,5))]);
        disp(['A_i mean | X =  1        : ' num2str(Designs(d,6))]);
        disp(['Pr(X=1)                  : ' num2str(Designs(d,1))]);
        disp('_________________________________________________________________________');
        disp('- NETWORK SUMMARY STATISTICS                                            -');
        disp('_________________________________________________________________________');
        disp(['Average density          : ' num2str(mean(MC_Results_design(:,1)))]);
        disp(['Average degree           : ' num2str(mean(MC_Results_design(:,2)))]);
        disp(['Average degree StdDev    : ' num2str(mean(MC_Results_design(:,4)))]);
        disp(['Average degree Skewness  : ' num2str(mean(MC_Results_design(:,5)))]);
        disp(['Average min degree       : ' num2str(mean(MC_Results_design(:,6)))]);
        disp(['Average max degree       : ' num2str(mean(MC_Results_design(:,7)))]);
       
        disp('_____________________________________________________');
        disp('- Convergence Statistics                            -');
        disp('_____________________________________________________');
        disp(['% ML estimates successfully computed  : ' num2str(100*mean(MC_Results_ml(:,end)))]);
        disp(['% PL estimates successfully computed  : ' num2str(100*mean(MC_Results_pl(:,end-1)))]);
        disp(['% JML estimates successfully computed : ' num2str(100*mean(MC_Results_jfe(:,end-1)))]);
        disp(['# NFX point iterations in final eval  : ' num2str(mean(MC_Results_jfe(:,end)))]);
        disp('');
        disp('_________________________________________________________');
        disp('- RMSE of ML, PL, JML and BC Estimators                 -');
        disp('_________________________________________________________');
        disp(['RMSE beta_ML         : ' num2str(rmse_ML)]);
        disp(['RMSE beta_PL         : ' num2str(rmse_PL)]);
        disp(['RMSE beta_JML        : ' num2str(rmse_JML)]);
        disp(['RMSE beta_BC         : ' num2str(rmse_BC)]);
        disp('_________________________________________________________');
        disp('- Mean and Median Bias of ML, PL, JML and BC Estimators -');
        disp('_________________________________________________________');
        disp(['beta0                : ' num2str(median(beta0))]);
        disp(['mean beta_ML         : ' num2str(mean(beta_ML(b_ML)))]);
        disp(['median beta_ML       : ' num2str(median(beta_ML(b_ML)))]);
        disp(['mean beta_PL         : ' num2str(mean(beta_PL(b_PL)))]);
        disp(['median beta_PL       : ' num2str(median(beta_PL(b_PL)))]);
        disp(['mean beta_JML        : ' num2str(mean(beta_JML(b_JML)))]);
        disp(['median beta_JML      : ' num2str(median(beta_JML(b_JML)))]);
        disp(['mean beta_BC         : ' num2str(mean(beta_BC(b_JML)))]);
        disp(['median beta_BC       : ' num2str(median(beta_BC(b_JML)))]);
        disp('');
        disp('_________________________________________________________');
        disp('- StdDev & StdErr of ML, PL, JML and BC Estimators      -');
        disp('_________________________________________________________');
        disp(['StdDev beta_ML           : ' num2str(std(beta_ML(b_ML)))]);
        disp(['median StdErr beta_ML    : ' num2str(median(MC_Results_ml(b_ML,2*K+1:3*K)))]);
        disp(['mean StdErr beta_ML      : ' num2str(mean(MC_Results_ml(b_ML,2*K+1:3*K)))]);
        disp(['StdDev beta_PL           : ' num2str(std(beta_PL(b_PL)))]);
        disp(['median StdErr beta_PL    : ' num2str(median(MC_Results_pl(b_PL,2*K+1:3*K)))]);
        disp(['mean StdErr beta_PL      : ' num2str(mean(MC_Results_pl(b_PL,2*K+1:3*K)))]);
        disp(['StdDev beta_JML          : ' num2str(std(beta_JML(b_JML)))]);
        disp(['StdDev beta_BC           : ' num2str(std(beta_BC(b_JML)))]);
        disp(['median StdErr beta_JML/BC: ' num2str(median(MC_Results_jfe(b_JML,3*K+1:4*K)))]);
        disp(['mean StdErr beta_JML/BC  : ' num2str(mean(MC_Results_jfe(b_JML,3*K+1:4*K)))]);
        disp('');
        disp('_____________________________________________________');
        disp('- Actual Size of alpha = 0.05 T-test                -');
        disp('_____________________________________________________');
        disp(['T-Test based beta_ML     : ' num2str(mean(MC_Results_ml(b_ML,3*K+1:4*K)))]);
        disp(['T-Test based beta_PL     : ' num2str(mean(MC_Results_pl(b_PL,3*K+1:4*K)))]);
        disp(['T-Test based beta_JML    : ' num2str(mean(MC_Results_jfe(b_JML,4*K+1:5*K)))]);
        disp(['T-Test based beta_BC     : ' num2str(mean(MC_Results_jfe(b_JML,5*K+1:6*K)))]);
        disp('');
        disp('_____________________________________________________');
        disp('- Actual Size of alpha = 0.10 T-test                -');
        disp('_____________________________________________________');
        disp(['T-Test based beta_ML     : ' num2str(mean(MC_Results_ml(b_ML,4*K+1:5*K)))]);
        disp(['T-Test based beta_PL     : ' num2str(mean(MC_Results_pl(b_PL,4*K+1:5*K)))]);
        disp(['T-Test based beta_JML    : ' num2str(mean(MC_Results_jfe(b_JML,6*K+1:7*K)))]);
        disp(['T-Test based beta_BC     : ' num2str(mean(MC_Results_jfe(b_JML,7*K+1:8*K)))]);
        diary off;
end

end

% Close parallel pool
delete(ParallelPool);



%------------------------------------------------------------%
%- Plot true A and JML A_hat for for first four designs     -%
%------------------------------------------------------------%

% Choose design
figure;

for d = 1:4
    % Set random number seed
    rng(361);
    pX          = Designs(d,1);
    AShape1     = Designs(d,2);
    AShape2     = Designs(d,3);
    ASuppLgth   = Designs(d,4);
    AMean0      = Designs(d,5);
    AMean1      = Designs(d,6);
    beta        = Designs(d,7);
    minA        = floor(2*(AMean0 - AShape1/(AShape1+AShape2)));
    maxA        = ceil(2*(AMean1 + AShape2/(AShape1+AShape2)));

    % STEP 1: Generate covariates and link data
    X_i    = 2*(random('bino',ones(N,1),pX*ones(N,1))-1/2);     
    W_ij   = repmat(X_i,1,N) .* repmat(X_i',N,1) - eye(N);
    W      = squareform(W_ij)';
    A_i    = AMean0*(X_i==-1) + AMean1*(X_i==1) ...
           + ASuppLgth*(random('beta',AShape1*ones(N,1),AShape2*ones(N,1)) - AShape1/(AShape1+AShape2));          
    A_ij   = repmat(A_i,1,N) + repmat(A_i',N,1) - 2*diag(A_i);
    A      = squareform(A_ij)';
    p      = exp(W*beta + A) ./ (1 + exp(W*beta + A));
    U      = random('unif',zeros(0.5*N*(N-1),1),ones(0.5*N*(N-1),1));    
    D      = (U<=p); 
    D_ij   = squareform(D);                                           

    % STEP 2: Plot JMLs of A_i's
    [beta_hat_jfe, bias_hat_jfe, A_i_hat] = betaSNM_JointFixedEffects(beta_sv, A_i_sv, D_ij, W, T, tol_NFP, MaxIter_NFP, silent, iterate, obs_hs);
    subplot(2,2,d)
    plot(A_i,A_i_hat,'o')
    line((minA/2):0.5:(maxA/2),(minA/2):0.5:(maxA/2));
    set(gca,'YTick',(minA/2):0.5:(maxA/2));
    %set(gca,'YTickLabel',{'-2','-3/2','-1','1/2','0','1/2','1'});
    set(gca,'XTick',(minA/2):0.5:(maxA/2));
    %set(gca,'XTickLabel',{'-2','-3/2','-1','1/2','0','1/2','1'});
    title('Joint FE');
end
         

