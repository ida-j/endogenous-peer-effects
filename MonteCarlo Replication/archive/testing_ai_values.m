clear all;
clc
%delete(gcp('nocreate'))
addpath(genpath('~/Dropbox/Ida (1)/REStat Third Round/MonteCarlo/'))

% cd('~/Dropbox/Ida (1)/REStat Third Round/MonteCarlo/')
cd ('~/Dropbox/Ida (1)/REStat Third Round/Revision Drafts/MonteCarloJuly2019/')

tstart = tic;

filename='Design1_exp';

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
% Designs = [ 
%             0.5     1       1       1   -0.5       -0.5          1 
%             0.5     1       1       1      -0.5          0.5     1;
%             0.5     1       1       1      0          0          1;
%             0.5     1       1       1   -0.25       -0.25        1;
%             0.5     1/4     3/4     1   -0.5         0           1;
%             0.5     1/4     3/4     1   -2/3        -1/6         1;
%             0.5     1/4     3/4     1   -0.75        -0.75       1;
%             0.5     1/4     3/4     1   -0.5          -0.5       1   
%             ];
% 
%         
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


for d=1:size(Designs,1)
    
designStats = zeros(B,5);
%---------------------------------------------------------%
%- Set up Monte Carlo Data Generating Process # 1        -%
%---------------------------------------------------------%

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


designStats(b,1) = corr(A_i,X_i);

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


designStats(b,2) = mean(DegreeDis);
designStats(b,3) = mean(skewness(DegreeDis,0));
designStats(b,4) = min(A_i);
designStats(b,5) = max(A_i);

% disp(std(DegreeDis))
% disp(skewness(DegreeDis))

end
disp(d);
disp(mean(designStats));
end

FID = fopen(filename, 'w');
fprintf(FID, '\\begin{table}\\caption{\\footnotesize {\\bf Design %.0f dense: Parameter values across %.0f Monte Carlo replications with $K_N=%.0f$}} \n',d,B,Kn);
fprintf(FID,'\\begin{threeparttable} \n');
fprintf(FID, '\\centering \\footnotesize\n');
fprintf(FID, '\\scalebox{.8}{\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\\toprule \n');

%% exp
fprintf(FID,'\\multicolumn{14}{c}{$h(a_i) = \\exp(a_i)$}\\\\ \n');
fprintf(FID,'\\cellcolor{yellow}$N$&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\\\\\hline \n',N1,N2);

fprintf(FID,'CF&$(0)$&$(1)$&$(2)$&$(3)$&$(4)$&$(5)$& $(0)$ &$(1)$&$(2)$&$(3)$&$(4)$&$(5)$&\\\\\\hline \n');


fprintf(FID,'\\multirow{4}{*}{$\\beta_1=%.0f$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f & %.3f& %.3f & \\textit{mean bias} \\\\ \n',b1,mat1_exp(1,1),mat1_exp(1,2),mat1_exp(1,3),mat1_exp(1,4),mat1_exp(1,5),mat1_exp(1,6),mat1_exp(1,7),mat1_exp(1,8),mat1_exp(1,9),mat1_exp(1,10),mat1_exp(1,11),mat1_exp(1,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_exp(2,1),mat1_exp(2,2),mat1_exp(2,3),mat1_exp(2,4),mat1_exp(2,5),mat1_exp(2,6),mat1_exp(2,7),mat1_exp(2,8),mat1_exp(2,9),mat1_exp(2,10),mat1_exp(2,11),mat1_exp(2,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{size} \\\\ \\midrule\n',mat1_exp(3,1),mat1_exp(3,2),mat1_exp(3,3),mat1_exp(3,4),mat1_exp(3,5),mat1_exp(3,6),mat1_exp(3,7),mat1_exp(3,8),mat1_exp(3,9),mat1_exp(3,10),mat1_exp(3,11),mat1_exp(3,12));


fprintf(FID,'\\multirow{4}{*}{$\\beta_2=5$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1_exp(5,1),mat1_exp(5,2),mat1_exp(5,3),mat1_exp(5,4),mat1_exp(5,5),mat1_exp(5,6),mat1_exp(5,7),mat1_exp(5,8),mat1_exp(5,9),mat1_exp(5,10),mat1_exp(5,11),mat1_exp(5,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_exp(6,1),mat1_exp(6,2),mat1_exp(6,3),mat1_exp(6,4),mat1_exp(6,5),mat1_exp(6,6),mat1_exp(6,7),mat1_exp(6,8),mat1_exp(6,9),mat1_exp(6,10),mat1_exp(6,11),mat1_exp(6,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1_exp(7,1),mat1_exp(7,2),mat1_exp(7,3),mat1_exp(7,4),mat1_exp(7,5),mat1_exp(7,6),mat1_exp(7,7),mat1_exp(7,8),mat1_exp(7,9),mat1_exp(7,10),mat1_exp(7,11),mat1_exp(7,12));


fprintf(FID,'\\multirow{4}{*}{$\\beta_3=5$}& %.3f & %.3f &%.3f& %.3f &%.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1_exp(9,1),mat1_exp(9,2),mat1_exp(9,3),mat1_exp(9,4),mat1_exp(9,5),mat1_exp(9,6),mat1_exp(9,7),mat1_exp(9,8),mat1_exp(9,9),mat1_exp(9,10),mat1_exp(9,11),mat1_exp(9,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_exp(10,1),mat1_exp(10,2),mat1_exp(10,3),mat1_exp(10,4),mat1_exp(10,5),mat1_exp(10,6),mat1_exp(10,7),mat1_exp(10,8),mat1_exp(10,9),mat1_exp(10,10),mat1_exp(10,11),mat1_exp(10,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f& %.3f &%.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1_exp(11,1),mat1_exp(11,2),mat1_exp(11,3),mat1_exp(11,4),mat1_exp(11,5),mat1_exp(11,6),mat1_exp(11,7),mat1_exp(11,8),mat1_exp(11,9),mat1_exp(11,10),mat1_exp(11,11),mat1_exp(11,12));

%% sin
fprintf(FID,'\\multicolumn{14}{c}{$h(a_i) = \\sin(a_i)$}\\\\ \n');
fprintf(FID,'\\cellcolor{yellow}$N$&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\\\\\hline \n',N1,N2);

fprintf(FID,'CF&$(0)$&$(1)$&$(2)$&$(3)$&$(4)$&$(5)$& $(0)$ &$(1)$&$(2)$&$(3)$&$(4)$&$(5)$&\\\\\\hline \n');


fprintf(FID,'\\multirow{4}{*}{$\\beta_1=%.0f$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f & %.3f& %.3f & \\textit{mean bias} \\\\ \n',b1,mat1_sin(1,1),mat1_sin(1,2),mat1_sin(1,3),mat1_sin(1,4),mat1_sin(1,5),mat1_sin(1,6),mat1_sin(1,7),mat1_sin(1,8),mat1_sin(1,9),mat1_sin(1,10),mat1_sin(1,11),mat1_sin(1,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_sin(2,1),mat1_sin(2,2),mat1_sin(2,3),mat1_sin(2,4),mat1_sin(2,5),mat1_sin(2,6),mat1_sin(2,7),mat1_sin(2,8),mat1_sin(2,9),mat1_sin(2,10),mat1_sin(2,11),mat1_sin(2,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{size} \\\\ \\midrule\n',mat1_sin(3,1),mat1_sin(3,2),mat1_sin(3,3),mat1_sin(3,4),mat1_sin(3,5),mat1_sin(3,6),mat1_sin(3,7),mat1_sin(3,8),mat1_sin(3,9),mat1_sin(3,10),mat1_sin(3,11),mat1_sin(3,12));


fprintf(FID,'\\multirow{4}{*}{$\\beta_2=5$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1_sin(5,1),mat1_sin(5,2),mat1_sin(5,3),mat1_sin(5,4),mat1_sin(5,5),mat1_sin(5,6),mat1_sin(5,7),mat1_sin(5,8),mat1_sin(5,9),mat1_sin(5,10),mat1_sin(5,11),mat1_sin(5,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_sin(6,1),mat1_sin(6,2),mat1_sin(6,3),mat1_sin(6,4),mat1_sin(6,5),mat1_sin(6,6),mat1_sin(6,7),mat1_sin(6,8),mat1_sin(6,9),mat1_sin(6,10),mat1_sin(6,11),mat1_sin(6,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1_sin(7,1),mat1_sin(7,2),mat1_sin(7,3),mat1_sin(7,4),mat1_sin(7,5),mat1_sin(7,6),mat1_sin(7,7),mat1_sin(7,8),mat1_sin(7,9),mat1_sin(7,10),mat1_sin(7,11),mat1_sin(7,12));


fprintf(FID,'\\multirow{4}{*}{$\\beta_3=5$}& %.3f & %.3f &%.3f& %.3f &%.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1_sin(9,1),mat1_sin(9,2),mat1_sin(9,3),mat1_sin(9,4),mat1_sin(9,5),mat1_sin(9,6),mat1_sin(9,7),mat1_sin(9,8),mat1_sin(9,9),mat1_sin(9,10),mat1_sin(9,11),mat1_sin(9,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_sin(10,1),mat1_sin(10,2),mat1_sin(10,3),mat1_sin(10,4),mat1_sin(10,5),mat1_sin(10,6),mat1_sin(10,7),mat1_sin(10,8),mat1_sin(10,9),mat1_sin(10,10),mat1_sin(10,11),mat1_sin(10,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f& %.3f &%.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1_sin(11,1),mat1_sin(11,2),mat1_sin(11,3),mat1_sin(11,4),mat1_sin(11,5),mat1_sin(11,6),mat1_sin(11,7),mat1_sin(11,8),mat1_sin(11,9),mat1_sin(11,10),mat1_sin(11,11),mat1_sin(11,12));

%% cos
fprintf(FID,'\\multicolumn{14}{c}{$h(a_i) = \\cos(a_i)$}\\\\ \n');
fprintf(FID,'\\cellcolor{yellow}$N$&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\\\\\hline \n',N1,N2);

fprintf(FID,'CF&$(0)$&$(1)$&$(2)$&$(3)$&$(4)$&$(5)$& $(0)$ &$(1)$&$(2)$&$(3)$&$(4)$&$(5)$&\\\\\\hline \n');


fprintf(FID,'\\multirow{4}{*}{$\\beta_1=%.0f$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f & %.3f& %.3f & \\textit{mean bias} \\\\ \n',b1,mat1_cos(1,1),mat1_cos(1,2),mat1_cos(1,3),mat1_cos(1,4),mat1_cos(1,5),mat1_cos(1,6),mat1_cos(1,7),mat1_cos(1,8),mat1_cos(1,9),mat1_cos(1,10),mat1_cos(1,11),mat1_cos(1,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_cos(2,1),mat1_cos(2,2),mat1_cos(2,3),mat1_cos(2,4),mat1_cos(2,5),mat1_cos(2,6),mat1_cos(2,7),mat1_cos(2,8),mat1_cos(2,9),mat1_cos(2,10),mat1_cos(2,11),mat1_cos(2,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{size} \\\\ \\midrule\n',mat1_cos(3,1),mat1_cos(3,2),mat1_cos(3,3),mat1_cos(3,4),mat1_cos(3,5),mat1_cos(3,6),mat1_cos(3,7),mat1_cos(3,8),mat1_cos(3,9),mat1_cos(3,10),mat1_cos(3,11),mat1_cos(3,12));


fprintf(FID,'\\multirow{4}{*}{$\\beta_2=5$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1_cos(5,1),mat1_cos(5,2),mat1_cos(5,3),mat1_cos(5,4),mat1_cos(5,5),mat1_cos(5,6),mat1_cos(5,7),mat1_cos(5,8),mat1_cos(5,9),mat1_cos(5,10),mat1_cos(5,11),mat1_cos(5,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_cos(6,1),mat1_cos(6,2),mat1_cos(6,3),mat1_cos(6,4),mat1_cos(6,5),mat1_cos(6,6),mat1_cos(6,7),mat1_cos(6,8),mat1_cos(6,9),mat1_cos(6,10),mat1_cos(6,11),mat1_cos(6,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1_cos(7,1),mat1_cos(7,2),mat1_cos(7,3),mat1_cos(7,4),mat1_cos(7,5),mat1_cos(7,6),mat1_cos(7,7),mat1_cos(7,8),mat1_cos(7,9),mat1_cos(7,10),mat1_cos(7,11),mat1_cos(7,12));


fprintf(FID,'\\multirow{4}{*}{$\\beta_3=5$}& %.3f & %.3f &%.3f& %.3f &%.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1_cos(9,1),mat1_cos(9,2),mat1_cos(9,3),mat1_cos(9,4),mat1_cos(9,5),mat1_cos(9,6),mat1_cos(9,7),mat1_cos(9,8),mat1_cos(9,9),mat1_cos(9,10),mat1_cos(9,11),mat1_cos(9,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1_cos(10,1),mat1_cos(10,2),mat1_cos(10,3),mat1_cos(10,4),mat1_cos(10,5),mat1_cos(10,6),mat1_cos(10,7),mat1_cos(10,8),mat1_cos(10,9),mat1_cos(10,10),mat1_cos(10,11),mat1_cos(10,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f& %.3f &%.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1_cos(11,1),mat1_cos(11,2),mat1_cos(11,3),mat1_cos(11,4),mat1_cos(11,5),mat1_cos(11,6),mat1_cos(11,7),mat1_cos(11,8),mat1_cos(11,9),mat1_cos(11,10),mat1_cos(11,11),mat1_cos(11,12));



fprintf(FID,'\\end{tabular}} \n');
%tablenotes
fprintf(FID,'\\begin{tablenotes}\\tiny \n');
fprintf(FID,'\\item CF - control function. $(0)$ - none, $(1)$ - $\\hat{a}_i$,  $(2)$ - $\\hat{h}(\\hat{a}_i)$, $(3)$ - $\\hat{h}(a_i)$, $(4)$ - $\\hat{h}(\\widehat{deg}_i,x_{2i})$, $(5)$ - $h(a_i)$. \n');
fprintf(FID,'\\item The network design parameters are $\\mu_0=%.2f$, $\\mu_1=%.2f$, $\\alpha_L=%.2f$, $\\alpha_H=%.2f$ \n',Designs(d,2),Designs(d,3),Designs(d,5),Designs(d,6));
fprintf(FID,'\\item Average number of links for $N=100$ is $%.1f$, for $N=250$ it is $%.1f$. \n',design_stats(1,2),design_stats(2,2));
fprintf(FID,'\\item Average skewness for $N=100$ is $%.2f$, for $N=250$ it is $%.2f$. \n',design_stats(1,5),design_stats(2,5));

fprintf(FID,'\\item Size is the empirical size of t-test against the truth. \n');
fprintf(FID,'\\item N$=100$, $corr(A_i,\\bf{x}_{2i})=%.3f$,N$=250$, $corr(A_i,\\bf{x}_{2i})=%.3f$ \n',corr1,corr2);

fprintf(FID,'\\item The bias of $\\hat{a}_i$ is calculated as $a_i-\\hat{a}_i$. \n'); 
fprintf(FID,'\\item For $N=100$, $\\hat{a}_i$ mean bias$=%.3f$, median bias$=%.3f$, std$=%.3f$. \n',Abias11(1), Abias11(2),Abias11(3));
fprintf(FID,'\\item For $N=250$, $\\hat{a}_i$ mean bias$=%.3f$, median bias$=%.3f$, std$=%.3f$. \n', Abias12(1), Abias12(2), Abias12(3));

fprintf(FID,'  \\end{tablenotes} \n');
fprintf(FID,'\\end{threeparttable} \n');
fprintf(FID,'\\end{table} \n');
fclose(FID);
            