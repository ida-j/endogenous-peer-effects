%acknowledgement:
%The part of the code that simulates the network is based on the replication code from a
% working paper version of “An econometric model of network formation with degree heterogeneity”
% by Bryan Graham


clear all;
clc
addpath(genpath('/MonteCarlo Replication/')) %path to the Monte Carlo Replication folder 
cd ('/tables/') % folder where tables are saved

tstart = tic;

filename='dense_network_designs.tex';

% hermite polynomial

Kn_values=[4 8];   % order of sieve
theta_y_values=[0.8] ;
sigma_e=1;
N=100;
kappa=3;

B = 1000; % Number of MC replications

theta_x=5;
theta_gx=5;
npar=3;

%Order of design parameters: 
%frequency of X = 1, mu0, mu1, l_x, mean of A_i (X=0)=alpha_L, mean of A_i (X=1)=alpha_H , lambda 
       
Designs = [ 
            0.5     1       1       1      -0.5        -0.5     1;
            0.5     1       1       1      0          0          1;
            0.5     1       1       1   -0.25       -0.25        1;
            0.5     1/4     3/4     1   -0.75        -0.75       1;
            0.5     1/4     3/4     1   -0.5         0           1;
            0.5     1/4     3/4     1   -2/3        -1/6         1;
            0.5     1/4     3/4       1   -0.5       0          1 
            0.5     1/4     3/4     1   -0.75          -0.5       1   
            ];
l_x = 1;                                     % Number of dyadic regressors 
 
% Optimmization parameters
lambda_sv     = zeros(l_x,1);              % Starting value for lambda_sv 
tol_NFP     = 1e-6;                        % Convergence criterion for fixed point iteration step 
MaxIter_NFP = 100;                         % Maximum number of NFP iteractions 
silent      = 1;                           % Show optimization output (or not) 
iterate     = 1;                           % Iterated bias correction     
obs_hs      = 1;                           % Used observed H_AA hessian instead of approximation for bias and variance estimation   

networkStats = zeros(size(Designs,1),5);
for d=1:size(Designs,1)
    
designStats = zeros(B,5);
%---------------------------------------------------------%
%- Set up Monte Carlo Data Generating Process # 1        -%
%---------------------------------------------------------%
rng(9);         % Set random number seed

% network
n = 0.5*N*(N-1);                           % Number of dyads     
  

% Compute 0.5N(N-1) x N matrix with T_ij terms
T = zeros(n,N);     % pre-allocate storage space for this matrix
for i = 1:(N-1)
    T(((n-(N-(i-1))*(N-i)/2) + 1):(n-(N-i)*(N-i-1)/2),:) = [zeros(N-i,i-1) ones(N-i,1) eye(N-i)];        
end

%-------------------------------------------------------------------%
%- Draw regressor matrix and heterogeneity parameters for design d -%
%-------------------------------------------------------------------%

pX          = Designs(d,1); % probability X=1
mu0         = Designs(d,2);
mu1         = Designs(d,3);
ASuppLgth   = Designs(d,4); 
alpha_L     = Designs(d,5);
alpha_H     = Designs(d,6);
lambda      = Designs(d,7);  


for b = 1:B

%-----------------------------------------------------%
%-        #1: Generate Network                       -%
%-----------------------------------------------------%   
% Draw observed agent-specific covariate: X = -1 or 1
X_i    = 2*(random('bino',ones(N,1),pX*ones(N,1))-1/2);     

X_ij   = repmat(X_i,1,N) + repmat(X_i',N,1)  - 2*diag(X_i);
X      = squareform(X_ij)';

% From W matrix (0.5N(N-1) X l_x) 
W_ij   = repmat(X_i,1,N) .* repmat(X_i',N,1) - eye(N);           % N x N matrix with dyad-specific regressor (interaction)
% W_ij   = abs(repmat(X_i,1,N) - repmat(X_i',N,1))-5 - eye(N).*diag((abs(repmat(X_i,1,N) - repmat(X_i',N,1))-5));

W      = squareform(W_ij)';                                      % 0.5N(N-1) X 1 vector with dyad-specific regressor

% Draw actor-specific heterogeneity
A_i = alpha_L*(X_i==-1) + alpha_H*(X_i==1)+ ASuppLgth*(random('beta',mu0*ones(N,1),mu1*ones(N,1)) - mu0/(mu0+mu1)); 



% form 0.5N(N-1) X 1 vector with A_i + A_j terms
A_ij = repmat(A_i,1,N) + repmat(A_i',N,1) - 2*diag(A_i);
A    = squareform(A_ij)';

% 0.5N(N-1) X 1 vector with ij link probabilities
p    = exp(W*lambda + A) ./ (1 + exp(W*lambda + A));

% Take random draw from network model for current design
U = random('unif',zeros(0.5*N*(N-1),1),ones(0.5*N*(N-1),1));    % 0.5N(N-1) X 1 vector of [0,1] uniforms
D = (U<=p); 
D_ij = squareform(D);
DegreeDis = sum(D_ij);

designStats(b,1) = corr(A_i,X_i);
designStats(b,2) = mean(DegreeDis);
designStats(b,3) = mean(skewness(DegreeDis,0));
designStats(b,4) = min(A_i);
designStats(b,5) = max(A_i);


end
disp(d);
disp(mean(designStats));
networkStats(d,:) = mean(designStats);
end
FID = fopen(filename, 'w');
fprintf(FID, '\\begin{table}[!h]\\caption{\\footnotesize {\\bf Statistics for dense network designs}} \\label{table: dense network design}\n');
fprintf(FID,'\\begin{threeparttable} \n');
fprintf(FID, '\\centering \\footnotesize\n');
fprintf(FID, '\\scalebox{1}{\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\\toprule \n');
fprintf(FID,'Design &\\bf 1& \\bf 2 & \\bf 3 & \\bf 4& \\bf 5 & \\bf 6 & \\bf 7 & \\bf 8 \\\\ \\midrule \n');
fprintf(FID,'$\\mu_0$ & %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f \\\\ \n',Designs(1,2),Designs(2,2),Designs(3,2),Designs(4,2),Designs(5,2),Designs(6,2),Designs(7,2),Designs(8,2));
fprintf(FID,'$\\mu_1$ & %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f \\\\ \n',Designs(1,3),Designs(2,3),Designs(3,3),Designs(4,3),Designs(5,3),Designs(6,3),Designs(7,3),Designs(8,3));
fprintf(FID,'$\\alpha_L$ & %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f \\\\ \n',Designs(1,5),Designs(2,5),Designs(3,5),Designs(4,5),Designs(5,5),Designs(6,5),Designs(7,5),Designs(8,5));
fprintf(FID,'$\\alpha_H$ & %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f \\\\ \n',Designs(1,6),Designs(2,6),Designs(3,6),Designs(4,6),Designs(5,6),Designs(6,6),Designs(7,6),Designs(8,6));
fprintf(FID,'$corr(a_i,\\bm{x}_{2i})$ & %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f \\\\ \n',networkStats(1,1),networkStats(2,1),networkStats(3,1),networkStats(4,1),networkStats(5,1),networkStats(6,1),networkStats(7,1),networkStats(8,1));
fprintf(FID,'Avg. Degree & %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f \\\\ \n',  networkStats(1,2),networkStats(2,2),networkStats(3,2),networkStats(4,2),networkStats(5,2),networkStats(6,2),networkStats(7,2),networkStats(8,2));
fprintf(FID,'Avg. Skewness & %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f& %.2f \\\\ \\bottomrule \n',networkStats(1,3),networkStats(2,3),networkStats(3,3),networkStats(4,3),networkStats(5,3),networkStats(6,3),networkStats(7,3),networkStats(8,3));

%% exp

fprintf(FID,'\\end{tabular}} \n');
%tablenotes
% fprintf(FID,'\\begin{tablenotes}\\tiny \n');
% fprintf(FID,'\\item Desings 1-4 , 5-8 \n');
% fprintf(FID,'  \\end{tablenotes} \n');
fprintf(FID,'\\end{threeparttable} \n');
fprintf(FID,'\\end{table} \n');
fclose(FID);
            