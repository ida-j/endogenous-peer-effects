clear all;
clc
%delete(gcp('nocreate'))
addpath(genpath('~/Dropbox/Ida (1)/REStat Third Round/MonteCarlo/'))

% cd('~/Dropbox/Ida (1)/REStat Third Round/MonteCarlo/')
cd ('~/Dropbox/Ida (1)/REStat Third Round/Revision Drafts/MonteCarloJuly2019/')



filename='Design1_exp';

% hermite polynomial

Kn_values=[4];   % order of sieve
theta_y_values=[0.8] ;
sigma_e=1;
N_values=[100 250];
kappa=3;

B = 1000; % Number of MC replications

theta_x=5;
theta_gx=5;
npar=3;

% Order of design parameters: 
%frequency of X = 1, mu0, mu1, l_x, mean of A_i (X=0)=alpha_L, mean of A_i (X=1)=alpha_H , lambda 
Designs = [ 0.5     1       1       1   0            0           1;
            0.5     1       1       1   -0.25       -0.25        1;
            0.5     1       1       1   -0.75       -0.75        1; %3
            0.5     1       1       1   -1.25       -1.25        1; %4
            0.5     1/4     3/4     1   0            0.5         1; %5
            0.5     1/4     3/4     1   -0.5         0           1;
            0.5     1/4     3/4     1   -1          -0.5         1;%7
            0.5     1/4     3/4     1   -1.5        -1           1;%8
            0.5     1/4     3/4     1   -0.75        -0.75       1];
l_x = 1;                                     % Number of dyadic regressors 
 
% Optimmization parameters
lambda_sv     = zeros(l_x,1);              % Starting value for lambda_sv 
tol_NFP     = 1e-6;                        % Convergence criterion for fixed point iteration step 
MaxIter_NFP = 100;                         % Maximum number of NFP iteractions 
silent      = 1;                           % Show optimization output (or not) 
iterate     = 1;                           % Iterated bias correction     
obs_hs      = 1;                           % Used observed H_AA hessian instead of approximation for bias and variance estimation   

results=cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));
sizes = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));

% for d=1:size(Designs,1)
for d = 1
for Kn_val=1:size(Kn_values,2)  
    for theta_y_val=1:size(theta_y_values,2)
            for N_val=1:size(N_values,2)

                Kn=Kn_values(1,Kn_val);
                theta_y=theta_y_values(1,theta_y_val);
                N=N_values(1,N_val);
                

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

%---------------------------------------------------------%
%-        Run Monte Carlo Experiment                     -%
%---------------------------------------------------------%

MC_Results_design   = zeros(B,7);                                          % Storage matrix for design features

rng(9);         % Set random number seed

% store estimates
estimates_true=zeros(B,3); 
estimates0=zeros(B,3);
estiamtes_lc=zeros(B,3);
estimates=zeros(B,3);
estimates_ahat=zeros(B,3);
estiamtes2=zeros(B,3);
MC_Results_design = zeros(B,7);
Amin = zeros(B,1);
Amax = zeros(B,1);
mnA = zeros(B,1);
medA = zeros(B,1);
stdA = zeros(B,1);


%ParallelPool = parpool;   % Open up a parallel pool
test = zeros(3,6);

vars_b = zeros(3,6);
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
W      = squareform(W_ij)';                                      % 0.5N(N-1) X 1 vector with dyad-specific regressor

% Draw actor-specific heterogeneity
A_i = alpha_L*(X_i==-1) + alpha_H*(X_i==1)+ ASuppLgth*(random('beta',mu0*ones(N,1),mu1*ones(N,1)) - mu0/(mu0+mu1)); 

% make sure A_i lies between -1 and 1
% if max(abs(A_i))>1
%     A_i=A_i/(max(abs(A_i))+0.001);
%     if b==1
%         disp(d)
%     end
% 
% end

% form 0.5N(N-1) X 1 vector with A_i + A_j terms
A_ij = repmat(A_i,1,N) + repmat(A_i',N,1) - 2*diag(A_i);
A    = squareform(A_ij)';

% 0.5N(N-1) X 1 vector with ij link probabilities
p    = exp(W*lambda + A) ./ (1 + exp(W*lambda + A));

% Take random draw from network model for current design
U = random('unif',zeros(0.5*N*(N-1),1),ones(0.5*N*(N-1),1));    % 0.5N(N-1) X 1 vector of [0,1] uniforms
D = (U<=p); 
D_ij = squareform(D);                                           % N x N adjacency matrix

%--------------------------------------------------%
%- # 2: Generate outcomes                         -%
%--------------------------------------------------%

G=normr(double(D_ij));
q1 = normrnd(X_i,1); q2 = normrnd(X_i,1); e = normrnd(zeros(N,1),1);

X2= 3*q1+cos(q2)/0.8+e;

H=cos(kappa*A_i);
Y=inv(eye(N)-theta_y*G)*(X2*theta_x+G*X2*theta_gx+H+normrnd(0,sigma_e,N,1));

%----------------------------------------------------%
%- # 3: Compute joint MLE estimates of lambda and A -%
%----------------------------------------------------%            
A_i_sv      = zeros(N,1);    % Starting values for A_i vector 

%[lambda_hat_jfe, bias_hat_jfe, A_i_hat_jfe, VCOV_hat_jfe, exitflag, NumFPIter] = betaSNM_JointFixedEffects(lambda_sv, A_i_sv, D_ij, W, T, tol_NFP, MaxIter_NFP, silent, iterate, obs_hs);
%----------------------------------------------------%
%- # 4: Estimate outcome equation parameters        -%
%----------------------------------------------------%   
A_i_hat_jfe = A_i;
W=[G*Y X2 G*X2]; % covariates
Z=[X2 G*X2 G^2*X2]; % instruments
                
            
%% sieve with A_hat

Qhat=zeros(N,Kn+1);
for ind=1:N
a=A_i_hat_jfe(ind);
Hp=[2*a; 4*a^2-2; 8*a^3-12*a; 16*a^4-48*a^2+12; 32*a^5-160*a^3+120*a; 64*a^6-480*a^4+720*a^2-120;128*a^7-1344*a^5+3360*a^3-1680*a; 256*a^8-3584*a^6+13440*a^4-13440*a^2+1680]*exp(-a^2/2);

for k=1:Kn 
Qhat(ind,k)=Hp(k);       
end
Qhat(ind,Kn+1)=1;
end
            
%% sieve with A_i

Q=zeros(N,Kn+1);
for ind=1:N
a=A_i(ind);
Hp=[2*a; 4*a^2-2; 8*a^3-12*a; 16*a^4-48*a^2+12; 32*a^5-160*a^3+120*a; 64*a^6-480*a^4+720*a^2-120;128*a^7-1344*a^5+3360*a^3-1680*a; 256*a^8-3584*a^6+13440*a^4-13440*a^2+1680]*exp(-a^2/2);

for k=1:Kn 
Q(ind,k)=Hp(k); 
end
Q(ind,Kn+1)=1;
end

%% sieve with deg_i and x_2i        
deg_dist=sum(D_ij)/(N-1);

Q2=zeros(N,2*Kn+1);
for ind=1:N
a=deg_dist(ind);
Hp=[2*a; 4*a^2-2; 8*a^3-12*a; 16*a^4-48*a^2+12; 32*a^5-160*a^3+120*a; 64*a^6-480*a^4+720*a^2-120;128*a^7-1344*a^5+3360*a^3-1680*a; 256*a^8-3584*a^6+13440*a^4-13440*a^2+1680]*exp(-a^2/2);

x=X_i(ind);
if x== -1
for k=1:Kn  
Q2(ind,k)=Hp(k);    
end
else
for k=Kn+1:2*Kn  
Q2(ind,k)=Hp(k-Kn);    
end 
end
Q2(ind,2*Kn+1)=1;          
end
           
           
%% Projection matrices          

% true H
M_H=eye(N)-H*inv(H'*H)*H';
% linear control
M_a=eye(N)-A_i_hat_jfe*inv(A_i_hat_jfe'*A_i_hat_jfe)*A_i_hat_jfe'; 
% sieve with a_i_hat
Mhat=eye(N)-Qhat*pinv(Qhat'*Qhat)*Qhat';
% sieve with a_i
M=eye(N)-Q*pinv(Q'*Q)*Q';
% sieve with deg_i and x_2i
M2=eye(N)-Q2*pinv(Q2'*Q2)*Q2';

%% true H estimator
thetahat_H=inv(W'*M_H*Z*inv(Z'*M_H*Z)*Z'*M_H*W)*W'*M_H*Z*inv(Z'*M_H*Z)*Z'*M_H*Y;
ehatH=M_H*(Y-W*thetahat_H);
VH = inv(W'*M_H*Z*inv(Z'*M_H*Z)*Z'*M_H*W)*sum(ehatH.^2)/N;

%% no control
thetahat0=inv(W'*Z*inv(Z'*Z)*Z'*W)*W'*Z*inv(Z'*Z)*Z'*Y;

ehat0=Y-W*thetahat0;
S0=0;
for j=1:N
S0=S0+Z(j,:)'*Z(j,:)*ehat0(j)^2; 
end
S0=S0/N;
% V0=inv(W'*Z*inv(N*S0)*Z'*W);
V0 = inv(W'*Z*inv(Z'*Z)*Z'*W)*sum(ehat0.^2)/N;

%% linear control
thetahat_lc=inv(W'*M_a*Z*inv(Z'*M_a*Z)*Z'*M_a*W)*W'*M_a*Z*inv(Z'*M_a*Z)*Z'*M_a*Y;
ehat_lc=M_a*(Y-W*thetahat_lc);
V_lc = inv(W'*M_a*Z*inv(Z'*M_a*Z)*Z'*M_a*W)*sum(ehat_lc.^2)/N;

%% a_i
thetahat=inv(W'*M*Z*inv(Z'*M*Z)*Z'*M*W)*W'*M*Z*inv(Z'*M*Z)*Z'*M*Y;
ehat=M*(Y-W*thetahat);
V = inv(W'*M*Z*inv(Z'*M*Z)*Z'*M*W)*sum(ehat.^2)/N;

%% a_i_hat
thetahat_ahat=inv(W'*Mhat*Z*inv(Z'*Mhat*Z)*Z'*Mhat*W)*W'*Mhat*Z*inv(Z'*Mhat*Z)*Z'*Mhat*Y;
ehat_ahat=Mhat*(Y-W*thetahat_ahat);
V_ahat = inv(W'*Mhat*Z*inv(Z'*Mhat*Z)*Z'*Mhat*W)*sum(ehat_ahat.^2)/N;
 
%% deg_i and x_2i
thetahat2=inv(W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*W)*W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*Y;         
ehat2=M2*(Y-W*thetahat2);
V2 = inv(W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*W)*sum(ehat2.^2)/N;

 
%% testing
null = [thetahat0 thetahat_lc thetahat_ahat thetahat thetahat2 thetahat_H]-[theta_y theta_y theta_y theta_y theta_y theta_y; theta_x theta_x theta_x theta_x theta_x theta_x; theta_gx theta_gx theta_gx theta_gx theta_gx theta_gx];
vars=[sqrt(diag(V0)) sqrt(diag(V_lc)) sqrt(diag(V_ahat)) sqrt(diag(V)) sqrt(diag(V2)) sqrt(diag(VH))];
vars_b = [vars_b;vars];
tst = null./vars;
test = test + double(abs(tst)>1.96);
%%
%----------------------------------------------------%
%- # 5: Store estimates                             -%
%----------------------------------------------------%   
DegreeDis = sum(D_ij);
MC_Results_design(b,:)  = [mean(D) mean(DegreeDis) median(DegreeDis) std(DegreeDis) skewness(DegreeDis,0) min(DegreeDis) max(DegreeDis)];

estimates0(b,:)=thetahat0';
estimates_lc(b,:)=thetahat_lc';
estimates_true(b,:)=thetahat_H';
estimates_ahat(b,:)=thetahat_ahat';
estimates(b,:)=thetahat';
estimates2(b,:)=thetahat2';

Amin(b,1)=min(A_i); Amax(b,1)=max(A_i);
mnA(b,1)=mean(A_i-A_i_hat_jfe);
medA(b,1)=median(A_i-A_i_hat_jfe);
stdA(b,1) = std(A_i-A_i_hat_jfe);

end  %% of B MC iterations

test = test/B;

%---------------------------------------------------------%
%- Store Monte Carlo results                             -%
%---------------------------------------------------------%
all_estimates=[estimates0 estimates_lc estimates_ahat estimates estimates2 estimates_true]; % B x (6x3=18)

design_av=[mean(MC_Results_design) min(Amin) max(Amax) mean(mnA) mean(medA) mean(stdA) zeros(1,6)];

results(d,Kn_val,theta_y_val,N_val)={[all_estimates;design_av]};
sizes(d,Kn_val,theta_y_val,N_val)={test};
                end % N
        end %theta_y
    end  %Kn





N1=N_values(1); N2=N_values(2);

for Kn_val=1:size(Kn_values,2)  

Kn=Kn_values(1,Kn_val);
sig=10*sigma_e;

filename=['MC_K_' num2str(Kn) '_d' num2str(d)  'design2_cos_hermite_polynomial.tex'];
b1=theta_y_values(1,1); 

%% results for N1 and N2

r11=results{d,Kn_val,1,1}(1:B,:); r12=results{d,Kn_val,1,2}(1:B,:);

t11 = sizes{d,Kn_val,1,1}; t12 = sizes{d,Kn_val,1,2};

% mean bias, std, median bias, interquartile range

mean11=mean(r11); std11=std(r11); median11=median(r11);  iqr11=iqr(r11);
mean12=mean(r12); std12=std(r12); median12=median(r12);  iqr12=iqr(r12);

% N1
mat11=[(mean11(1)-b1)           (mean11(4)-b1)           (mean11(7)-b1)         (mean11(10)-b1)     (mean11(13)-b1)      (mean11(16)-b1);  % beta
        std11(1)                 std11(4)                std11(7)               std11(10)           std11(13)            std11(16) ;
        t11(1,1)                t11(1,2)                t11(1,3)                t11(1,4)            t11(1,5)             t11(1,6) ;
        iqr11(1)                iqr11(4)                iqr11(7)                iqr11(10)           iqr11(13)            iqr11(16)       ;

        (mean11(2)-theta_x)    (mean11(5)-theta_x)    (mean11(8)-theta_x)     (mean11(11)-theta_x)  (mean11(14)-theta_x)  (mean11(17)-theta_x)           ; 
        std11(2)                std11(5)                std11(8)                std11(11)           std11(14)            std11(17) ;
        t11(2,1)                t11(2,2)                t11(2,3)                t11(2,4)            t11(2,5)             t11(2,6);
        iqr11(2)                iqr11(5)                iqr11(8)                iqr11(11)           iqr11(14)            iqr11(17);

        (mean11(3)-theta_x)    (mean11(6)-theta_x)     (mean11(9)-theta_x)     (mean11(12)-theta_x) (mean11(15)-theta_x)  (mean11(18)-theta_x) ; 
        std11(3)                std11(6)                std11(9)                std11(12)           std11(15)              std11(18) ;
        t11(3,1)                t11(3,2)                t11(3,3)                t11(3,4)            t11(3,5)               t11(3,6) ;
        iqr11(3)                iqr11(6)                iqr11(9)                iqr11(12)           iqr11(15)              iqr11(18)];
            
% N2
mat12=[(mean12(1)-b1)           (mean12(4)-b1)           (mean12(7)-b1)         (mean12(10)-b1)     (mean12(13)-b1)      (mean12(16)-b1);  % beta
        std12(1)                 std12(4)                std12(7)               std12(10)           std12(13)            std12(16) ;
        t12(1,1)                t12(1,2)                t12(1,3)                t12(1,4)            t12(1,5)             t12(1,6) ;
        iqr12(1)                iqr12(4)                iqr12(7)                iqr12(10)           iqr12(13)            iqr12(16)       ;

        (mean12(2)-theta_x)    (mean12(5)-theta_x)    (mean12(8)-theta_x)     (mean12(11)-theta_x)  (mean12(14)-theta_x)  (mean12(17)-theta_x)           ; 
        std12(2)                std12(5)                std12(8)                std12(11)           std12(14)            std12(17) ;
        t12(2,1)                t12(2,2)                t12(2,3)                t12(2,4)            t12(2,5)             t12(2,6);
        iqr12(2)                iqr12(5)                iqr12(8)                iqr12(11)           iqr12(14)            iqr12(17);

        (mean12(3)-theta_x)    (mean12(6)-theta_x)     (mean12(9)-theta_x)     (mean12(12)-theta_x) (mean12(15)-theta_x)  (mean12(18)-theta_x) ; 
        std12(3)                std12(6)                std12(9)                std12(12)           std12(15)              std12(18) ;
        t12(3,1)                t12(3,2)                t12(3,3)                t12(3,4)            t12(3,5)               t12(3,6) ;
        iqr12(3)                iqr12(6)                iqr12(9)                iqr12(12)           iqr12(15)              iqr12(18)];
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mat1=[mat11 mat12];

Abias11=[results{d,Kn_val,1,1}(B+1,10:12)];
Abias12=[results{d,Kn_val,1,2}(B+1,10:12)];
    

design_stats=[results{d,Kn_val,1,1}(B+1,1:9);
results{d,Kn_val,1,2}(B+1,1:9);];
des=mean(design_stats);
des=des(1:7);

amin=min(design_stats(:,8)); amax=max(design_stats(:,9));

FID = fopen(filename, 'w');
fprintf(FID, '\\begin{table}\\caption{\\footnotesize {\\bf Design $%.0f$}: Parameter values across %.0f Monte Carlo replications with $h(a)=\\cos (3 a_i)$ and $K_N=%.0f$} \n',d,B,Kn);
fprintf(FID,'\\begin{threeparttable} \n');
fprintf(FID, '\\centering \\footnotesize\n');
fprintf(FID, '\\scalebox{.9}{\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\\hline \n');

%% b1

fprintf(FID,'\\cellcolor{yellow}$N$&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\\\\\hline \n',N1,N2);

fprintf(FID,'CF&$(0)$&$(1)$&$(2)$&$(3)$&$(4)$&$(5)$& $(0)$ &$(1)$&$(2)$&$(3)$&$(4)$&$(5)$&\\\\\\hline \n');

% beta1

fprintf(FID,'\\multirow{4}{*}{$\\beta_1=%.0f$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f & %.3f& %.3f & \\textit{mean bias} \\\\ \n',b1,mat1(1,1),mat1(1,2),mat1(1,3),mat1(1,4),mat1(1,5),mat1(1,6),mat1(1,7),mat1(1,8),mat1(1,9),mat1(1,10),mat1(1,11),mat1(1,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1(2,1),mat1(2,2),mat1(2,3),mat1(2,4),mat1(2,5),mat1(2,6),mat1(2,7),mat1(2,8),mat1(2,9),mat1(2,10),mat1(2,11),mat1(2,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{size} \\\\ \\midrule\n',mat1(3,1),mat1(3,2),mat1(3,3),mat1(3,4),mat1(3,5),mat1(3,6),mat1(3,7),mat1(3,8),mat1(3,9),mat1(3,10),mat1(3,11),mat1(3,12));


% beta2

fprintf(FID,'\\multirow{4}{*}{$\\beta_2=5$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1(5,1),mat1(5,2),mat1(5,3),mat1(5,4),mat1(5,5),mat1(5,6),mat1(5,7),mat1(5,8),mat1(5,9),mat1(5,10),mat1(5,11),mat1(5,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1(6,1),mat1(6,2),mat1(6,3),mat1(6,4),mat1(6,5),mat1(6,6),mat1(6,7),mat1(6,8),mat1(6,9),mat1(6,10),mat1(6,11),mat1(6,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1(7,1),mat1(7,2),mat1(7,3),mat1(7,4),mat1(7,5),mat1(7,6),mat1(7,7),mat1(7,8),mat1(7,9),mat1(7,10),mat1(7,11),mat1(7,12));


% beta3
 

fprintf(FID,'\\multirow{4}{*}{$\\beta_3=5$}& %.3f & %.3f &%.3f& %.3f &%.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1(9,1),mat1(9,2),mat1(9,3),mat1(9,4),mat1(9,5),mat1(9,6),mat1(9,7),mat1(9,8),mat1(9,9),mat1(9,10),mat1(9,11),mat1(9,12));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1(10,1),mat1(10,2),mat1(10,3),mat1(10,4),mat1(10,5),mat1(10,6),mat1(10,7),mat1(10,8),mat1(10,9),mat1(10,10),mat1(10,11),mat1(10,12));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f& %.3f &%.3f &%.3f & %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1(11,1),mat1(11,2),mat1(11,3),mat1(11,4),mat1(11,5),mat1(11,6),mat1(11,7),mat1(11,8),mat1(11,9),mat1(11,10),mat1(11,11),mat1(11,12));




fprintf(FID,'\\end{tabular}} \n');
%tablenotes
fprintf(FID,'\\begin{tablenotes}\\tiny \n');
fprintf(FID,'\\item CF - control function. $(0)$ - none, $(1)$ - $\\hat{a}_i$,  $(2)$ - $\\hat{h}(\\hat{a}_i)$, $(3)$ - $\\hat{h}(a_i)$, $(4)$ - $\\hat{h}(\\widehat{deg}_i,x_{2i})$, $(5)$ - $h(a_i)$. \n');
fprintf(FID,'\\item Average number of links for $N=100$ is $%.1f$, for $N=250$ it is $%.1f$. \n',design_stats(1,2),design_stats(2,2));
fprintf(FID,'\\item Average skewness for $N=100$ is $%.1f$, for $N=250$ it is $%.1f$. \n',design_stats(1,5),design_stats(2,5));

fprintf(FID,'\\item Size is the empirical size of t-test against the truth. \n');
fprintf(FID,'\\item The bias of $\\hat{a}_i$ is calculated as $a_i-\\hat{a}_i$.'); 
fprintf(FID,'For $N=100$, $\\hat{a}_i$ mean bias$=%.3f$, median bias$=%.3f$, std$=%.3f$.',Abias11(1), Abias11(2),Abias11(3));
fprintf(FID,'For $N=250$, $\\hat{a}_i$ mean bias$=%.3f$, median bias$=%.3f$, std$=%.3f$.', Abias12(1), Abias12(2), Abias12(3));

fprintf(FID,'  \\end{tablenotes} \n');
fprintf(FID,'\\end{threeparttable} \n');
fprintf(FID,'\\end{table} \n');
fclose(FID);
            
            
   end
 
end



