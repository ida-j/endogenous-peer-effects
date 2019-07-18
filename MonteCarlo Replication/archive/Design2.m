clear all;
clc
delete(gcp('nocreate'))
addpath(genpath('~/Dropbox/Ida (1)/REStat Second Round/MonteCarlo/'))

cd('~/Dropbox/Ida (1)/REStat Second Round/MonteCarlo/')

% hermite plynomial sieve
% h(a)=exp(a)/(kappa+exp(a))

tstart = tic;

filename='Design1';


Kn_values=[4];   % order of sieve
theta_y_values=[0.2 0.5 0.8] ;
sigma_e=1;
N_values=[100 250];
kappa=3;

B = 500;                                  % Number of MC replications

theta_x=1;
theta_gx=1;
npar=3;
 

% Order of design parameters: 
%frequency of X = 1, mu0, mu1, l_x, mean of A_i (X=0)=alpha_L, mean of A_i (X=1)=alpha_H , lambda 
Designs = [ 0.5     1       1       1   0            0           1;
            0.5     1       1       1   -0.25       -0.25        1;
            0.5     1       1       1   -0.75       -0.75        1;
            0.5     1       1       1   -1.25       -1.25        1;
            0.5     1/4     3/4     1   0            0.5         1;
            0.5     1/4     3/4     1   -0.5         0           1;
            0.5     1/4     3/4     1   -1          -0.5         1;...
            0.5     1/4     3/4     1   -1.5        -1           1];
l_x = 1;                                     % Number of dyadic regressors 
 
% Optimmization parameters
lambda_sv     = zeros(l_x,1);                  % Starting value for lambda_sv 
tol_NFP     = 1e-6;                        % Convergence criterion for fixed point iteration step 
MaxIter_NFP = 100;                         % Maximum number of NFP iteractions 
silent      = 1;                           % Show optimization output (or not) 
iterate     = 1;                           % Iterated bias correction     
obs_hs      = 1;                           % Used observed H_AA hessian instead of approximation for bias and variance estimation   

results=cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));
sizes = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));

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




d=8;    
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


% estimates_true=zeros(B,3);
% estimates0=zeros(B,3);
% estiamtes_lc=zeros(B,3);
% estimates=zeros(B,3);
% estiamtes2=zeros(B,3);



%ParallelPool = parpool;   % Open up a parallel pool
test = zeros(3,5);

vars_b = zeros(3,5);
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
            A_i = alpha_L*(X_i==-1) + alpha_H*(X_i==1) ...
                + ASuppLgth*(random('beta',mu0*ones(N,1),mu1*ones(N,1)) - mu0/(mu0+mu1)); 
            
%             % make sure A_i lies between -1 and 1
            if max(abs(A_i))>1
                A_i=A_i/(max(abs(A_i))+0.001);
            end

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
            % 3q1+cos(q2)/0.8+e_i where q1,q2 ~ N(X_i,1) and e_i~N(0,1)
            q1 = normrnd(X_i,1); q2 = normrnd(X_i,1); e = normrnd(zeros(N,1),1);
            
            X2= 3*q1+cos(q2)/0.8+e;

            H=sin(kappa* A_i);
            Y=inv(eye(N)-theta_y*G)*(X2*theta_x+G*X2*theta_gx+H+normrnd(0,sigma_e,N,1));
        
        
            %----------------------------------------------------%
            %- # 3: Compute joint MLE estimates of lambda and A -%
            %----------------------------------------------------%
            
            A_i_sv      = zeros(N,1);    % Starting values for A_i vector 

            [lambda_hat_jfe, bias_hat_jfe, A_i_hat_jfe, VCOV_hat_jfe, exitflag, NumFPIter] = betaSNM_JointFixedEffects(lambda_sv, A_i_sv, D_ij, W, T, tol_NFP, MaxIter_NFP, silent, iterate, obs_hs);

            %----------------------------------------------------%
            %- # 4: Estimate outcome equation parameters        -%
            %----------------------------------------------------%   
  
            W=[G*Y X2 G*X2]; % covariates
            Z=[X2 G*X2 G^2*X2]; % instruments
                
            
%% sieve
            
              % sieve basis
            Q=zeros(N,Kn+1);
            for ind=1:N
             a=A_i_hat_jfe(ind);

                for k=1:Kn  
               Q(ind,k)=a^k;    

                end
                Q(ind,Kn+1)=1;
            end
      
%% sieve with deg_i and x_2i        
            deg_dist=sum(D_ij)/(N-1);

               Q2=zeros(N,2*Kn+1);
           for ind=1:N
              a=deg_dist(ind);
              x=X_i(ind);
                  
              if x== -1
                for k=1:Kn  
               Q2(ind,k)=a^k;    
                end
              else
                for k=Kn+1:2*Kn  
              Q2(ind,k)=a^(k-Kn);    
    
                end 
              end
              
           Q2(ind,2*Kn+1)=1;          
           end
           
%% Estimators           

% true H
M_H=eye(N)-H*inv(H'*H)*H';

% linear control
M_a=eye(N)-A_i_hat_jfe*inv(A_i_hat_jfe'*A_i_hat_jfe)*A_i_hat_jfe'; 

% sieve with a_i
M=eye(N)-Q*inv(Q'*Q)*Q';

% sieve with deg_i and x_2i
M2=eye(N)-Q2*pinv(Q2'*Q2)*Q2';

%% true H
thetahat_H=inv(W'*M_H*Z*inv(Z'*M_H*Z)*Z'*M_H*W)*W'*M_H*Z*inv(Z'*M_H*Z)*Z'*M_H*Y;
ehatH=Y-W*thetahat_H;
VH = inv(W'*M_H*Z*inv(Z'*M_H*Z)*Z'*M_H*W)*sum(ehatH.^2)/N;

 

%% no control
thetahat0=inv(W'*Z*inv(Z'*Z)*Z'*W)*W'*Z*inv(Z'*Z)*Z'*Y;

 ehat0=Y-W*thetahat0;
 S0=0;
 for j=1:N
    S0=S0+Z(j,:)'*Z(j,:)*ehat0(j)^2; 
 end
 S0=S0/N;
 V0=inv(W'*Z*inv(N*S0)*Z'*W);
 

%% linear control
thetahat_lc=inv(W'*M_a*Z*inv(Z'*M_a*Z)*Z'*M_a*W)*W'*M_a*Z*inv(Z'*M_a*Z)*Z'*M_a*Y;

ehat_lc=M_a*(Y-W*thetahat_lc);

 V_lc = inv(W'*M_a*Z*inv(Z'*M_a*Z)*Z'*M_a*W)*sum(ehat_lc.^2)/N;

%% a_i
thetahat=inv(W'*M*Z*inv(Z'*M*Z)*Z'*M*W)*W'*M*Z*inv(Z'*M*Z)*Z'*M*Y;

 
ehat=M*(Y-W*thetahat);
 

 
 V = inv(W'*M*Z*inv(Z'*M*Z)*Z'*M*W)*sum(ehat.^2)/N;

 
%% deg_i and x_2i
thetahat2=inv(W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*W)*W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*Y;
              
ehat2=M2*(Y-W*thetahat2);
 
V2 = inv(W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*W)*sum(ehat2.^2)/N;

 
%% testing
null = [thetahat0 thetahat_lc thetahat thetahat2 thetahat_H]-[theta_y theta_y theta_y theta_y theta_y; 1 1 1 1 1; 1 1 1 1 1];

vars=[sqrt(diag(V0)) sqrt(diag(V_lc)) sqrt(diag(V)) sqrt(diag(V2)) sqrt(diag(VH))];

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

            estimates(b,:)=thetahat';
            estimates2(b,:)=thetahat2';

            Amin(b,1)=min(A_i); Amax(b,1)=max(A_i);
            
              mnA(b,1)=mean(A_i-A_i_hat_jfe);
            medA(b,1)=median(A_i-A_i_hat_jfe);
         
end  %% of B MC iterations

test = test/B;


        
% Close parallel pool
%delete(ParallelPool);


      

        %---------------------------------------------------------%
        %- Store Monte Carlo results                             -%
        %---------------------------------------------------------%
        all_estimates=[estimates0 estimates_lc estimates estimates2 estimates_true]; % B x (5x3=15)

        design_av=[mean(MC_Results_design) min(Amin) max(Amax) mean(mnA) mean(medA) zeros(1,4)];
        
   results(d,Kn_val,theta_y_val,N_val)={[all_estimates;design_av]};
   sizes(d,Kn_val,theta_y_val,N_val)={test};


                end % N
        end %theta_y
    end  %Kn


save(filename,'results');

telapsed = toc(tstart);
disp(['elapsed time: ' num2str(telapsed/60) ' min']);


N1=N_values(1); N2=N_values(2);
d=8;

    for Kn_val=1:size(Kn_values,2)  

                Kn=Kn_values(1,Kn_val);
                sig=10*sigma_e;
                
       filename=['MC_K_' num2str(Kn) '_sig_' num2str(sig)  '_kappa_' num2str(kappa)  '_h2_polynomial.tex'];
% beta values 
b1=theta_y_values(1,1); b2=theta_y_values(1,2); b3=theta_y_values(1,3);

%% results for b=0.2, 0.5, 0.8 (rows) & kappa = 1, 10 (columns)

r11=results{d,Kn_val,1,1}(1:B,:);r12=results{d,Kn_val,1,2}(1:B,:);
r21=results{d,Kn_val,2,1}(1:B,:);r22=results{d,Kn_val,2,2}(1:B,:);
r31=results{d,Kn_val,3,1}(1:B,:);r32=results{d,Kn_val,3,2}(1:B,:);

t11 = sizes{d,Kn_val,1,1}; t12 = sizes{d,Kn_val,1,2};
t21 = sizes{d,Kn_val,2,1}; t22 = sizes{d,Kn_val,2,2};
t31 = sizes{d,Kn_val,3,1}; t32 = sizes{d,Kn_val,3,2};


% mean bias, std, median bias, interquartile range

%%%%%%%%%%%% beta1 %%%%%%%%%%%%%%%
mean11=mean(r11); std11=std(r11); median11=median(r11);  iqr11=iqr(r11);
mean12=mean(r12); std12=std(r12); median12=median(r12);  iqr12=iqr(r12);


 % kappa1  
mat11=[(mean11(1)-b1)    (mean11(4)-b1)     (mean11(7)-b1)     (mean11(10)-b1)    (mean11(13)-b1) ;  % beta
       std11(1)          std11(4)           std11(7)           std11(10)          std11(13)       ;
       t11(1,1)         t11(1,2)            t11(1,3)            t11(1,4)           t11(1,5)   ;
       iqr11(1)          iqr11(4)            iqr11(7)         iqr11(10)          iqr11(13)           ;
       
     (mean11(2)-1)    (mean11(5)-1)     (mean11(8)-1)     (mean11(11)-1)    (mean11(14)-1)    ; 
     std11(2)          std11(5)           std11(8)           std11(11)          std11(14)     ;
       t11(2,1)         t11(2,2)            t11(2,3)            t11(2,4)           t11(2,5)   ;
     iqr11(2)          iqr11(5)            iqr11(8)         iqr11(11)          iqr11(14)       ;
       
      (mean11(3)-1)    (mean11(6)-1)     (mean11(9)-1)     (mean11(12)-1)    (mean11(15)-1)     ; 
      std11(3)          std11(6)           std11(9)           std11(12)          std11(15)       ;
       t11(3,1)         t11(3,2)            t11(3,3)            t11(3,4)           t11(3,5)   ;
      iqr11(3)          iqr11(6)            iqr11(9)         iqr11(12)          iqr11(15)     ];
       
       
% kappa2       
mat12=[(mean12(1)-b1)    (mean12(4)-b1)     (mean12(7)-b1)     (mean12(10)-b1)    (mean12(13)-b1) ;  % beta
       std12(1)          std12(4)           std12(7)           std12(10)          std12(13)       ;
       t12(1,1)         t12(1,2)            t12(1,3)            t12(1,4)           t12(1,5)   ;
       iqr12(1)          iqr12(4)            iqr12(7)         iqr12(10)          iqr12(13)           ;
       
     (mean12(2)-1)    (mean12(5)-1)     (mean12(8)-1)     (mean12(11)-1)    (mean12(14)-1)    ; 
     std12(2)          std12(5)           std12(8)           std12(11)          std12(14)     ;
       t12(2,1)         t12(2,2)            t12(2,3)            t12(2,4)           t12(2,5)   ;
     iqr12(2)          iqr12(5)            iqr12(8)         iqr12(11)          iqr12(14)       ;
       
      (mean12(3)-1)    (mean12(6)-1)     (mean12(9)-1)     (mean12(12)-1)    (mean12(15)-1)     ; 
      std12(3)          std12(6)           std12(9)           std12(12)          std12(15)       ;
       t12(3,1)         t12(3,2)            t12(3,3)            t12(3,4)           t12(3,5)   ;
      iqr12(3)          iqr12(6)            iqr12(9)         iqr12(12)          iqr12(15)     ];

%%%%%%%%%%%% beta2 %%%%%%%%%%%%%%%


mean21=mean(r21); std21=std(r21); median21=median(r21); iqr21=iqr(r21);
mean22=mean(r22); std22=std(r22); median22=median(r22); iqr22=iqr(r22);

mat21=[(mean21(1)-b2)    (mean21(4)-b2)     (mean21(7)-b2)     (mean21(10)-b2)    (mean21(13)-b2) ;  % beta
       std21(1)          std21(4)           std21(7)           std21(10)          std21(13)       ;
       t21(1,1)         t21(1,2)            t21(1,3)            t21(1,4)           t21(1,5)   ;
       iqr21(1)          iqr21(4)            iqr21(7)         iqr21(10)          iqr21(13)           ;
       
     (mean21(2)-1)    (mean21(5)-1)     (mean21(8)-1)     (mean21(11)-1)    (mean21(14)-1)    ; 
     std21(2)          std21(5)           std21(8)           std21(11)          std21(14)     ;
       t21(2,1)         t21(2,2)            t21(2,3)            t21(2,4)           t21(2,5)   ;
     iqr21(2)          iqr21(5)            iqr21(8)         iqr21(11)          iqr21(14)       ;
       
      (mean21(3)-1)    (mean21(6)-1)     (mean21(9)-1)     (mean21(12)-1)    (mean21(15)-1)     ; 
      std21(3)          std21(6)           std21(9)           std21(12)          std21(15)       ;
       t21(3,1)         t21(3,2)            t21(3,3)            t21(3,4)           t21(3,5)   ;
      iqr21(3)          iqr21(6)            iqr21(9)         iqr21(12)          iqr21(15)     ];
       
       
% kappa2       
mat22=[(mean22(1)-b2)    (mean22(4)-b2)     (mean22(7)-b2)     (mean22(10)-b2)    (mean22(13)-b2) ;  % beta
       std22(1)          std22(4)           std22(7)           std22(10)          std22(13)       ;
       t22(1,1)         t22(1,2)            t22(1,3)            t22(1,4)           t22(1,5)   ;
       iqr22(1)          iqr22(4)            iqr22(7)         iqr22(10)          iqr22(13)           ;
       
     (mean22(2)-1)    (mean22(5)-1)     (mean22(8)-1)     (mean22(11)-1)    (mean22(14)-1)    ; 
     std22(2)          std22(5)           std22(8)           std22(11)          std22(14)     ;
       t22(2,1)         t22(2,2)            t22(2,3)            t22(2,4)           t22(2,5)   ;
     iqr22(2)          iqr22(5)            iqr22(8)         iqr22(11)          iqr22(14)       ;
       
      (mean22(3)-1)    (mean22(6)-1)     (mean22(9)-1)     (mean22(12)-1)    (mean22(15)-1)     ; 
      std22(3)          std22(6)           std22(9)           std22(12)          std22(15)       ;
       t22(3,1)         t22(3,2)            t22(3,3)            t22(3,4)           t22(3,5)   ;
      iqr22(3)          iqr22(6)            iqr22(9)         iqr22(12)          iqr22(15)     ];



   
%%%%%%%%%%%% beta3 %%%%%%%%%%%%%%%
   

mean31=mean(r31); std31=std(r31); median31=median(r31); iqr31=iqr(r31);
mean32=mean(r32); std32=std(r32); median32=median(r32); iqr32=iqr(r32);

mat31=[(mean31(1)-b3)    (mean31(4)-b3)     (mean31(7)-b3)     (mean31(10)-b3)    (mean31(13)-b3) ;  % beta
       std31(1)          std31(4)           std31(7)           std31(10)          std31(13)       ;
       t31(1,1)         t31(1,2)            t31(1,3)            t31(1,4)           t31(1,5)   ;
       iqr31(1)          iqr31(4)            iqr31(7)         iqr31(10)          iqr31(13)           ;
       
     (mean31(2)-1)    (mean31(5)-1)     (mean31(8)-1)     (mean31(11)-1)    (mean31(14)-1)    ; 
     std31(2)          std31(5)           std31(8)           std31(11)          std31(14)     ;
       t31(2,1)         t31(2,2)            t31(2,3)            t31(2,4)           t31(2,5)   ;
     iqr31(2)          iqr31(5)            iqr31(8)         iqr31(11)          iqr31(14)       ;
       
      (mean31(3)-1)    (mean31(6)-1)     (mean31(9)-1)     (mean31(12)-1)    (mean31(15)-1)     ; 
      std31(3)          std31(6)           std31(9)           std31(12)          std31(15)       ;
       t31(3,1)         t31(3,2)            t31(3,3)            t31(3,4)           t31(3,5)   ;
      iqr31(3)          iqr31(6)            iqr31(9)         iqr31(12)          iqr31(15)     ];
       
       
% kappa2       
mat32=[(mean32(1)-b3)    (mean32(4)-b3)     (mean32(7)-b3)     (mean32(10)-b3)    (mean32(13)-b3) ;  % beta
       std32(1)          std32(4)           std32(7)           std32(10)          std32(13)       ;
       t32(1,1)         t32(1,2)            t32(1,3)            t32(1,4)           t32(1,5)   ;
       iqr32(1)          iqr32(4)            iqr32(7)         iqr32(10)          iqr32(13)           ;
       
     (mean32(2)-1)    (mean32(5)-1)     (mean32(8)-1)     (mean32(11)-1)    (mean32(14)-1)    ; 
     std32(2)          std32(5)           std32(8)           std32(11)          std32(14)     ;
       t32(2,1)         t32(2,2)            t32(2,3)            t32(2,4)           t32(2,5)   ;
     iqr32(2)          iqr32(5)            iqr32(8)         iqr32(11)          iqr32(14)       ;
       
      (mean32(3)-1)    (mean32(6)-1)     (mean32(9)-1)     (mean32(12)-1)    (mean32(15)-1)     ; 
      std32(3)          std32(6)           std32(9)           std32(12)          std32(15)       ;
       t32(3,1)         t32(3,2)            t32(3,3)            t32(3,4)           t32(3,5)   ;
      iqr32(3)          iqr32(6)            iqr32(9)         iqr32(12)          iqr32(15)     ];
       


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mat1=[mat11 mat12];
mat2=[mat21 mat22];
mat3=[mat31 mat32];

Abias11=[results{d,Kn_val,1,1}(B+1,10:11)];
Abias21=[results{d,Kn_val,2,1}(B+1,10:11)];
Abias31=[results{d,Kn_val,3,1}(B+1,10:11)];
Abias12=[results{d,Kn_val,1,2}(B+1,10:11)];
Abias22=[results{d,Kn_val,2,2}(B+1,10:11)];
Abias32=[results{d,Kn_val,3,2}(B+1,10:11)];
    

design_stats=[results{d,Kn_val,1,1}(B+1,1:9);
results{d,Kn_val,2,1}(B+1,1:9);
results{d,Kn_val,3,1}(B+1,1:9);
results{d,Kn_val,1,2}(B+1,1:9);
results{d,Kn_val,2,2}(B+1,1:9);
results{d,Kn_val,3,2}(B+1,1:9);];
des=mean(design_stats);
des=des(1:7);

amin=min(design_stats(:,8)); amax=max(design_stats(:,9));



FID = fopen(filename, 'w');
fprintf(FID, '\\begin{table}\\caption{\\footnotesize {\\bf Polynomial Sieve}: Parameter values across %.0f Monte Carlo replications with $h(a)=\\exp (3 a_i)$ and $K_N=%.0f$} \n',B,Kn);
fprintf(FID,'\\begin{threeparttable} \n');
fprintf(FID, '\\centering \\footnotesize\n');
fprintf(FID, '\\scalebox{.9}{\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\\hline \n');

%% b1

fprintf(FID,'\\cellcolor{yellow}$N$&\\multicolumn{5}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{5}{|c|}{\\cellcolor{yellow}$%.0f$}&\\\\\\hline \n',N1,N2);

fprintf(FID,'CF&$(0)$&$(1)$&$(2)$&$(3)$&$(4)$& $(0)$ &$(1)$&$(2)$&$(3)$&$(4)$&\\\\\\hline \n');

% beta

fprintf(FID,'\\multirow{4}{*}{$\\beta_1=%.0f$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',b1,mat1(1,1),mat1(1,2),mat1(1,3),mat1(1,4),mat1(1,5),mat1(1,6),mat1(1,7),mat1(1,8),mat1(1,9),mat1(1,10));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1(2,1),mat1(2,2),mat1(2,3),mat1(2,4),mat1(2,5),mat1(2,6),mat1(2,7),mat1(2,8),mat1(2,9),mat1(2,10));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{size} \\\\ \\midrule\n',mat1(3,1),mat1(3,2),mat1(3,3),mat1(3,4),mat1(3,5),mat1(3,6),mat1(3,7),mat1(3,8),mat1(3,9),mat1(3,10));


% gamma

fprintf(FID,'\\multirow{4}{*}{$\\beta_2=1$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1(5,1),mat1(5,2),mat1(5,3),mat1(5,4),mat1(5,5),mat1(5,6),mat1(5,7),mat1(5,8),mat1(5,9),mat1(5,10));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1(6,1),mat1(6,2),mat1(6,3),mat1(6,4),mat1(6,5),mat1(6,6),mat1(6,7),mat1(6,8),mat1(6,9),mat1(6,10));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1(7,1),mat1(7,2),mat1(7,3),mat1(7,4),mat1(7,5),mat1(7,6),mat1(7,7),mat1(7,8),mat1(7,9),mat1(7,10));


% delta
 

fprintf(FID,'\\multirow{4}{*}{$\\beta_3=1$}& %.3f & %.3f &%.3f& %.3f &%.3f &%.3f &%.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat1(9,1),mat1(9,2),mat1(9,3),mat1(9,4),mat1(9,5),mat1(9,6),mat1(9,7),mat1(9,8),mat1(9,9),mat1(9,10));

fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat1(10,1),mat1(10,2),mat1(10,3),mat1(10,4),mat1(10,5),mat1(10,6),mat1(10,7),mat1(10,8),mat1(10,9),mat1(10,10));

fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f& %.3f &%.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat1(11,1),mat1(11,2),mat1(11,3),mat1(11,4),mat1(11,5),mat1(11,6),mat1(11,7),mat1(11,8),mat1(11,9),mat1(11,10));


fprintf(FID,'&\\multicolumn{5}{|c|}{$\\hat{a}$ - mean bias$=$%.3f, median bias$=$%.3f }&\\multicolumn{5}{|c|}{$\\hat{a}$ - mean bias$=$%.3f, median bias$=$%.3f }&\\\\ \\hline \n', Abias11(1), Abias11(2), Abias12(1), Abias12(2));

%% b2

fprintf(FID,'\\cellcolor{yellow}$N$&\\multicolumn{5}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{5}{|c|}{\\cellcolor{yellow}$%.0f$}&\\\\\\hline \n',N1,N2);

fprintf(FID,'CF&$(0)$&$(1)$&$(2)$&$(3)$&$(4)$& $(0)$ &$(1)$&$(2)$&$(3)$&$(4)$&\\\\\\hline \n');

% beta
 
fprintf(FID,'\\multirow{4}{*}{$\\beta_1=%.1f$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',b2,mat2(1,1),mat2(1,2),mat2(1,3),mat2(1,4),mat2(1,5),mat2(1,6),mat2(1,7),mat2(1,8),mat2(1,9),mat2(1,10));
 
fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat2(2,1),mat2(2,2),mat2(2,3),mat2(2,4),mat2(2,5),mat2(2,6),mat2(2,7),mat2(2,8),mat2(2,9),mat2(2,10));
 
fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{size} \\\\ \\midrule \n',mat2(3,1),mat2(3,2),mat2(3,3),mat2(3,4),mat2(3,5),mat2(3,6),mat2(3,7),mat2(3,8),mat2(3,9),mat2(3,10));
 
 
% gamma
fprintf(FID,'\\multirow{4}{*}{$\\beta_2=1$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat2(5,1),mat2(5,2),mat2(5,3),mat2(5,4),mat2(5,5),mat2(5,6),mat2(5,7),mat2(5,8),mat2(5,9),mat2(5,10));
 
fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat2(6,1),mat2(6,2),mat2(6,3),mat2(6,4),mat2(6,5),mat2(6,6),mat2(6,7),mat2(6,8),mat2(6,9),mat2(6,10));
 
fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat2(7,1),mat2(7,2),mat2(7,3),mat2(7,4),mat2(7,5),mat2(7,6),mat2(7,7),mat2(7,8),mat2(7,9),mat2(7,10));
 
 
% delta
 
 
fprintf(FID,'\\multirow{4}{*}{$\\beta_3=1$}& %.3f & %.3f &%.3f& %.3f &%.3f &%.3f &%.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat2(9,1),mat2(9,2),mat2(9,3),mat2(9,4),mat2(9,5),mat2(9,6),mat2(9,7),mat2(9,8),mat2(9,9),mat2(9,10));
 
fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat2(10,1),mat2(10,2),mat2(10,3),mat2(10,4),mat2(10,5),mat2(10,6),mat2(10,7),mat2(10,8),mat2(10,9),mat2(10,10));
 
fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f& %.3f &%.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat2(11,1),mat2(11,2),mat2(11,3),mat2(11,4),mat2(11,5),mat2(11,6),mat2(11,7),mat2(11,8),mat2(11,9),mat2(11,10));
 


fprintf(FID,'&\\multicolumn{5}{|c|}{$\\hat{a}$ - mean bias$=$%.3f, median bias$=$%.3f }&\\multicolumn{5}{|c|}{$\\hat{a}$ - mean bias$=$%.3f, median bias$=$%.3f }&\\\\ \\hline \n', Abias21(1), Abias21(2), Abias22(1), Abias22(2));

%% b3

fprintf(FID,'\\cellcolor{yellow}$N$&\\multicolumn{5}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{5}{|c|}{\\cellcolor{yellow}$%.0f$}&\\\\\\hline \n',N1,N2);

fprintf(FID,'CF&$(0)$&$(1)$&$(2)$&$(3)$&$(4)$& $(0)$ &$(1)$&$(2)$&$(3)$&$(4)$&\\\\\\hline \n');

% beta
 
fprintf(FID,'\\multirow{4}{*}{$\\beta_1=%.1f$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',b3,mat3(1,1),mat3(1,2),mat3(1,3),mat3(1,4),mat3(1,5),mat3(1,6),mat3(1,7),mat3(1,8),mat3(1,9),mat3(1,10));
 
fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat3(2,1),mat3(2,2),mat3(2,3),mat3(2,4),mat3(2,5),mat3(2,6),mat3(2,7),mat3(2,8),mat3(2,9),mat3(2,10));
 
fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{size} \\\\\\midrule \n',mat3(3,1),mat3(3,2),mat3(3,3),mat3(3,4),mat3(3,5),mat3(3,6),mat3(3,7),mat3(3,8),mat3(3,9),mat3(3,10));
 
 
% gamma
 
fprintf(FID,'\\multirow{4}{*}{$\\beta_2=1$}& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat3(5,1),mat3(5,2),mat3(5,3),mat3(5,4),mat3(5,5),mat3(5,6),mat3(5,7),mat3(5,8),mat3(5,9),mat3(5,10));
 
fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat3(6,1),mat3(6,2),mat3(6,3),mat3(6,4),mat3(6,5),mat3(6,6),mat3(6,7),mat3(6,8),mat3(6,9),mat3(6,10));
 
fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f &%.3f& %.3f &%.3f &\\textit{size} \\\\ \\midrule\n',mat3(7,1),mat3(7,2),mat3(7,3),mat3(7,4),mat3(7,5),mat3(7,6),mat3(7,7),mat3(7,8),mat3(7,9),mat3(7,10));
 
 
% delta
 
fprintf(FID,'\\multirow{4}{*}{$\\beta_3=1$}& %.3f & %.3f &%.3f& %.3f &%.3f &%.3f &%.3f &%.3f &%.3f& %.3f &\\textit{mean bias} \\\\ \n',mat3(9,1),mat3(9,2),mat3(9,3),mat3(9,4),mat3(9,5),mat3(9,6),mat3(9,7),mat3(9,8),mat3(9,9),mat3(9,10));
 
fprintf(FID,'&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&(%.3f )&\\textit{std}\\\\ \n',mat3(10,1),mat3(10,2),mat3(10,3),mat3(10,4),mat3(10,5),mat3(10,6),mat3(10,7),mat3(10,8),mat3(10,9),mat3(10,10));
 
fprintf(FID,'& %.3f & %.3f &%.3f &%.3f &%.3f &%.3f& %.3f& %.3f &%.3f &%.3f &\\textit{size} \\\\\\midrule \n',mat3(11,1),mat3(11,2),mat3(11,3),mat3(11,4),mat3(11,5),mat3(11,6),mat3(11,7),mat3(11,8),mat3(11,9),mat3(11,10));
 


fprintf(FID,'&\\multicolumn{5}{|c|}{$\\hat{a}$ - mean bias$=$%.3f, median bias$=$%.3f }&\\multicolumn{5}{|c|}{$\\hat{a}$ - mean bias$=$%.3f, median bias$=$%.3f }&\\\\ \\hline \n', Abias31(1), Abias31(2), Abias32(1), Abias32(2));

%%

fprintf(FID,'\\end{tabular}} \n');
%tablenotes
fprintf(FID,'\\begin{tablenotes}\\tiny \n');
fprintf(FID,'\\item CF - control function. $(0)$ - none, $(1)$ - $\\hat{a}_i$,  $(2)$ - $\\hat{h}(\\hat{a}_i)$, $(3)$ - $\\hat{h}(\\widehat{deg}_i,x_{2i})$, $(4)$ - $h(a_i)$. \n');
fprintf(FID,'\\item Average number of links for $N=100$ is $24.1$, for $N=250$ it is $60.2$. \n');
fprintf(FID,'\\item The bias of $\\hat{a}_i$ is calculated as $a_i-\\hat{a}_i$ \n');
fprintf(FID,'\\item $K^*_{100,0.2}=3$, $K^*_{100,0.5}=5$, $K^*_{100,0.8}=4$ \n');
fprintf(FID,'\\item $K^*_{250,0.2}=8$, $K^*_{250,0.5}=3$, $K^*_{250,0.8}=8$ \n');
fprintf(FID,'  \\end{tablenotes} \n');
fprintf(FID,'\\end{threeparttable} \n');
fprintf(FID,'\\end{table} \n');
fclose(FID);
            
   end
 

