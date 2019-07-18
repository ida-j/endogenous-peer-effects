%acknowledgement:
%The part of the code that simulates the network is based on the replication code from a
% working paper version of “An econometric model of network formation with degree heterogeneity”
% by Bryan Graham


clear all;
clc
addpath(genpath('/MonteCarlo Replication/')) %path to the Monte Carlo Replication folder 
cd ('/tables/') % folder where tables are saved


% Order of design parameters: 
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


Kn_values=[3 4 5 6 7 8];   % order of sieve
theta_y_values=[0.8] ;
sigma_e=1;
N_values=[100 250];
kappa=3;

B = 1000;                                  % Number of MC replications

theta_x=5;
theta_gx=5;
npar=3;

for d=1:size(Designs,1)
    
rmse_res_h0 = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));
rmse_res_deg0 = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));

rmse_res_h1 = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));
rmse_res_deg1 = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));

rmse_res_h2 = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));
rmse_res_deg2 = cell(size(Designs,1),size(Kn_values,2),size(theta_y_values,2),size(N_values,2));

for design = 0:2

switch design
    case 0
filename = 'Design1exp_Kn';
fname='Kn_Design1exptex';

    case 1
filename = 'Design2cos_Kn';
fname='Kn_Design2cos.tex';       
   
    case 2
filename = 'Design3sin_Kn';
fname='Design3sin.tex';

end


tstart = tic;


l_x = 1;                                     % Number of dyadic regressors 
 
% Optimmization parameters
lambda_sv     = zeros(l_x,1);                  % Starting value for lambda_sv 
tol_NFP     = 1e-6;                        % Convergence criterion for fixed point iteration step 
MaxIter_NFP = 100;                         % Maximum number of NFP iteractions 
silent      = 1;                           % Show optimization output (or not) 
iterate     = 1;                           % Iterated bias correction     
obs_hs      = 1;                           % Used observed H_AA hessian instead of approximation for bias and variance estimation   




    for Kn_val=1:size(Kn_values,2)  
        for theta_y_val=1:size(theta_y_values,2)
                for N_val=1:size(N_values,2)

                Kn=Kn_values(1,Kn_val);
                theta_y=theta_y_values(1,theta_y_val);
                N=N_values(1,N_val);
disp(d)               
disp(N)
disp(Kn)
disp(design)
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
        
        
    rmse_h = [];
    rmse_deg = [];
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
            q1 = normrnd(X_i,1); q2 = normrnd(X_i,1); e = normrnd(zeros(N,1),1);
            
            X2= 3*q1+cos(q2)/0.8+e;
switch design
    case 0
 H=exp(kappa* A_i);


    case 1
H=cos(kappa* A_i);      
   
    case 2
H=sin(kappa* A_i);

end


            YY=inv(eye(N)-theta_y*G)*(X2*theta_x+G*X2*theta_gx+H+normrnd(0,sigma_e,N,1));
        
        
            %----------------------------------------------------%
            %- # 3: Compute joint MLE estimates of lambda and A -%
            %----------------------------------------------------%
            
          %  A_i_sv      = zeros(N,1);    % Starting values for A_i vector 

          %  [lambda_hat_jfe, bias_hat_jfe, A_i_hat_jfe, VCOV_hat_jfe, exitflag, NumFPIter] = betaSNM_JointFixedEffects(lambda_sv, A_i_sv, D_ij, W, T, tol_NFP, MaxIter_NFP, silent, iterate, obs_hs);
            A_i_hat_jfe = A_i;
            %----------------------------------------------------%
            %- # 4: Estimate outcome equation parameters        -%
            %----------------------------------------------------%   
  
            WW=[G*YY X2 G*X2]; % covariates
            ZZ=[X2 G*X2 G^2*X2]; % instruments
                
 
%% sieve with h(a)

            QQ=zeros(N,Kn+1);
            for ind=1:N
             a=A_i_hat_jfe(ind);
               Hp=[2*a; 4*a^2-2; 8*a^3-12*a; 16*a^4-48*a^2+12; 32*a^5-160*a^3+120*a; 64*a^6-480*a^4+720*a^2-120;128*a^7-1344*a^5+3360*a^3-1680*a; 256*a^8-3584*a^6+13440*a^4-13440*a^2+1680]*exp(-a^2/2);

                for k=1:Kn 

               QQ(ind,k)=Hp(k); 
               
                end
                QQ(ind,Kn+1)=1;
            end
            
            %% sieve with deg_i and x_2i        
            deg_dist=sum(D_ij)/(N-1);

           
     QQ2=zeros(N,2*Kn+1);
           for ind=1:N
              a=deg_dist(ind);
              x=X_i(ind);
              Hp=[2*a; 4*a^2-2; 8*a^3-12*a; 16*a^4-48*a^2+12; 32*a^5-160*a^3+120*a; 64*a^6-480*a^4+720*a^2-120;128*a^7-1344*a^5+3360*a^3-1680*a; 256*a^8-3584*a^6+13440*a^4-13440*a^2+1680]*exp(-a^2/2);

              if x== -1
                for k=1:Kn  
            QQ2(ind,k)=Hp(k);  
           % QQ2(ind,k) = a^k;
                end
              else
                for k=Kn+1:2*Kn  
            QQ2(ind,k)=Hp(k-Kn);    
           % QQ2(ind,k)=a^(k-Kn);
                end 
              end
              
           QQ2(ind,2*Kn+1)=1;          
           end   

for n = 1:N
    
    yi = YY(n);
Wi = WW(n,:);
Zi = ZZ(n,:);   

Y = YY(1:end ~=n);
Z = ZZ(1:end ~= n,:);
W = WW(1:end ~= n,:);
Q = QQ(1:end ~=n,:);
Q2 = QQ2(1:end ~=n,:);


M2=eye(N-1)-Q2*pinv(Q2'*Q2)*Q2';

%% h(a)
M=eye(N-1)-Q*inv(Q'*Q)*Q';
thetahat=inv(W'*M*Z*inv(Z'*M*Z)*Z'*M*W)*W'*M*Z*inv(Z'*M*Z)*Z'*M*Y;

%% deg_i and x_2i
thetahat2=inv(W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*W)*W'*M2*Z*inv(Z'*M2*Z)*Z'*M2*Y;
              


rmse_h(end+1) = (yi-Wi*thetahat)^2;
rmse_deg(end+1) = (yi - Wi*thetahat2)^2;
end % end of CV leave-1-out iterations 
end  %% of B MC iterations

switch design
    case 0
      rmse_res_deg0(d,Kn_val,theta_y_val,N_val)={[mean(rmse_deg), median(rmse_deg), std(rmse_deg), iqr(rmse_deg)]};
      rmse_res_h0(d,Kn_val,theta_y_val,N_val)={[mean(rmse_h), median(rmse_h), std(rmse_h), iqr(rmse_h)]};

    case 1
      rmse_res_deg1(d,Kn_val,theta_y_val,N_val)={[mean(rmse_deg), median(rmse_deg), std(rmse_deg), iqr(rmse_deg)]};
      rmse_res_h1(d,Kn_val,theta_y_val,N_val)={[mean(rmse_h), median(rmse_h), std(rmse_h), iqr(rmse_h)]};  
   
    case 2
      rmse_res_deg2(d,Kn_val,theta_y_val,N_val)={[mean(rmse_deg), median(rmse_deg), std(rmse_deg), iqr(rmse_deg)]};
      rmse_res_h2(d,Kn_val,theta_y_val,N_val)={[mean(rmse_h), median(rmse_h), std(rmse_h), iqr(rmse_h)]};

end



                end % N
        end %theta_y
    end  %Kn


telapsed = toc(tstart);
disp(['elapsed time: ' num2str(telapsed/60) ' min']);


end
save('cv.mat', 'rmse_res_h0', 'rmse_res_h1', 'rmse_res_h2', 'rmse_res_deg0', 'rmse_res_deg1','rmse_res_deg2');
rmse_res_deg2(d,Kn_val,theta_y_val,N_val);


% mat0_100_h = [rmse_res_h0{8,1,1,1}(1,1) rmse_res_h0{8,2,1,1}(1,1) rmse_res_h0{8,3,1,1}(1,1)  rmse_res_h0{8,4,1,1}(1,1) rmse_res_h0{8,5,1,1}(1,1) rmse_res_h0{8,6,1,1}(1,1); 
% rmse_res_h0{8,1,1,1}(1,2) rmse_res_h0{8,2,1,1}(1,2) rmse_res_h0{8,3,1,1}(1,2)  rmse_res_h0{8,4,1,1}(1,2) rmse_res_h0{8,5,1,1}(1,2) rmse_res_h0{8,6,1,1}(1,2);
% rmse_res_h0{8,1,1,1}(1,3) rmse_res_h0{8,2,1,1}(1,3) rmse_res_h0{8,3,1,1}(1,3)  rmse_res_h0{8,4,1,1}(1,3) rmse_res_h0{8,5,1,1}(1,3) rmse_res_h0{8,6,1,1}(1,3);
% rmse_res_h0{8,1,1,1}(1,4) rmse_res_h0{8,2,1,1}(1,4) rmse_res_h0{8,3,1,1}(1,4)  rmse_res_h0{8,4,1,1}(1,4) rmse_res_h0{8,5,1,1}(1,4) rmse_res_h0{8,6,1,1}(1,4)];
% 
% mat1_100_h = [rmse_res_h1{8,1,1,1}(1,1) rmse_res_h1{8,2,1,1}(1,1) rmse_res_h1{8,3,1,1}(1,1)  rmse_res_h1{8,4,1,1}(1,1) rmse_res_h1{8,5,1,1}(1,1) rmse_res_h1{8,6,1,1}(1,1); 
% rmse_res_h1{8,1,1,1}(1,2) rmse_res_h1{8,2,1,1}(1,2) rmse_res_h1{8,3,1,1}(1,2)  rmse_res_h1{8,4,1,1}(1,2) rmse_res_h1{8,5,1,1}(1,2) rmse_res_h1{8,6,1,1}(1,2);
% rmse_res_h1{8,1,1,1}(1,3) rmse_res_h1{8,2,1,1}(1,3) rmse_res_h1{8,3,1,1}(1,3)  rmse_res_h1{8,4,1,1}(1,3) rmse_res_h1{8,5,1,1}(1,3) rmse_res_h1{8,6,1,1}(1,3);
% rmse_res_h1{8,1,1,1}(1,4) rmse_res_h1{8,2,1,1}(1,4) rmse_res_h1{8,3,1,1}(1,4)  rmse_res_h1{8,4,1,1}(1,4) rmse_res_h1{8,5,1,1}(1,4) rmse_res_h1{8,6,1,1}(1,4)];
% 
% mat2_100_h = [rmse_res_h2{8,1,1,1}(1,1) rmse_res_h2{8,2,1,1}(1,1) rmse_res_h2{8,3,1,1}(1,1)  rmse_res_h2{8,4,1,1}(1,1) rmse_res_h2{8,5,1,1}(1,1) rmse_res_h2{8,6,1,1}(1,1); 
% rmse_res_h2{8,1,1,1}(1,2) rmse_res_h2{8,2,1,1}(1,2) rmse_res_h2{8,3,1,1}(1,2)  rmse_res_h2{8,4,1,1}(1,2) rmse_res_h2{8,5,1,1}(1,2) rmse_res_h2{8,6,1,1}(1,2);
% rmse_res_h2{8,1,1,1}(1,3) rmse_res_h2{8,2,1,1}(1,3) rmse_res_h2{8,3,1,1}(1,3)  rmse_res_h2{8,4,1,1}(1,3) rmse_res_h2{8,5,1,1}(1,3) rmse_res_h2{8,6,1,1}(1,3);
% rmse_res_h2{8,1,1,1}(1,4) rmse_res_h2{8,2,1,1}(1,4) rmse_res_h2{8,3,1,1}(1,4)  rmse_res_h2{8,4,1,1}(1,4) rmse_res_h2{8,5,1,1}(1,4) rmse_res_h2{8,6,1,1}(1,4)];
% 
% mat0_250_h = [rmse_res_h0{8,1,1,2}(1,1) rmse_res_h0{8,2,1,2}(1,1) rmse_res_h0{8,3,1,2}(1,1)  rmse_res_h0{8,4,1,2}(1,1) rmse_res_h0{8,5,1,2}(1,1) rmse_res_h0{8,6,1,2}(1,1); 
% rmse_res_h0{8,1,1,2}(1,2) rmse_res_h0{8,2,1,2}(1,2) rmse_res_h0{8,3,1,2}(1,2)  rmse_res_h0{8,4,1,2}(1,2) rmse_res_h0{8,5,1,2}(1,2) rmse_res_h0{8,6,1,2}(1,2);
% rmse_res_h0{8,1,1,2}(1,3) rmse_res_h0{8,2,1,2}(1,3) rmse_res_h0{8,3,1,2}(1,3)  rmse_res_h0{8,4,1,2}(1,3) rmse_res_h0{8,5,1,2}(1,3) rmse_res_h0{8,6,1,2}(1,3);
% rmse_res_h0{8,1,1,2}(1,4) rmse_res_h0{8,2,1,2}(1,4) rmse_res_h0{8,3,1,2}(1,4)  rmse_res_h0{8,4,1,2}(1,4) rmse_res_h0{8,5,1,2}(1,4) rmse_res_h0{8,6,1,2}(1,4)];
% 
% mat1_250_h = [rmse_res_h1{8,1,1,2}(1,1) rmse_res_h1{8,2,1,2}(1,1) rmse_res_h1{8,3,1,2}(1,1)  rmse_res_h1{8,4,1,2}(1,1) rmse_res_h1{8,5,1,2}(1,1) rmse_res_h1{8,6,1,2}(1,1); 
% rmse_res_h1{8,1,1,2}(1,2) rmse_res_h1{8,2,1,2}(1,2) rmse_res_h1{8,3,1,2}(1,2)  rmse_res_h1{8,4,1,2}(1,2) rmse_res_h1{8,5,1,2}(1,2) rmse_res_h1{8,6,1,2}(1,2);
% rmse_res_h1{8,1,1,2}(1,3) rmse_res_h1{8,2,1,2}(1,3) rmse_res_h1{8,3,1,2}(1,3)  rmse_res_h1{8,4,1,2}(1,3) rmse_res_h1{8,5,1,2}(1,3) rmse_res_h1{8,6,1,2}(1,3);
% rmse_res_h1{8,1,1,2}(1,4) rmse_res_h1{8,2,1,2}(1,4) rmse_res_h1{8,3,1,2}(1,4)  rmse_res_h1{8,4,1,2}(1,4) rmse_res_h1{8,5,1,2}(1,4) rmse_res_h1{8,6,1,2}(1,4)];
% 
% mat2_250_h = [rmse_res_h2{8,1,1,2}(1,1) rmse_res_h2{8,2,1,2}(1,1) rmse_res_h2{8,3,1,2}(1,1)  rmse_res_h2{8,4,1,2}(1,1) rmse_res_h2{8,5,1,2}(1,1) rmse_res_h2{8,6,1,2}(1,1); 
% rmse_res_h2{8,1,1,2}(1,2) rmse_res_h2{8,2,1,2}(1,2) rmse_res_h2{8,3,1,2}(1,2)  rmse_res_h2{8,4,1,2}(1,2) rmse_res_h2{8,5,1,2}(1,2) rmse_res_h2{8,6,1,2}(1,2);
% rmse_res_h2{8,1,1,2}(1,3) rmse_res_h2{8,2,1,2}(1,3) rmse_res_h2{8,3,1,2}(1,3)  rmse_res_h2{8,4,1,2}(1,3) rmse_res_h2{8,5,1,2}(1,3) rmse_res_h2{8,6,1,2}(1,3);
% rmse_res_h2{8,1,1,2}(1,4) rmse_res_h2{8,2,1,2}(1,4) rmse_res_h2{8,3,1,2}(1,4)  rmse_res_h2{8,4,1,2}(1,4) rmse_res_h2{8,5,1,2}(1,4) rmse_res_h2{8,6,1,2}(1,4)];
% 
% %%deg
% 
% mat0_100_deg = [rmse_res_deg0{8,1,1,1}(1,1) rmse_res_deg0{8,2,1,1}(1,1) rmse_res_deg0{8,3,1,1}(1,1)  rmse_res_deg0{8,4,1,1}(1,1) rmse_res_deg0{8,5,1,1}(1,1) rmse_res_deg0{8,6,1,1}(1,1); 
% rmse_res_deg0{8,1,1,1}(1,2) rmse_res_deg0{8,2,1,1}(1,2) rmse_res_deg0{8,3,1,1}(1,2)  rmse_res_deg0{8,4,1,1}(1,2) rmse_res_deg0{8,5,1,1}(1,2) rmse_res_deg0{8,6,1,1}(1,2);
% rmse_res_deg0{8,1,1,1}(1,3) rmse_res_deg0{8,2,1,1}(1,3) rmse_res_deg0{8,3,1,1}(1,3)  rmse_res_deg0{8,4,1,1}(1,3) rmse_res_deg0{8,5,1,1}(1,3) rmse_res_deg0{8,6,1,1}(1,3);
% rmse_res_deg0{8,1,1,1}(1,4) rmse_res_deg0{8,2,1,1}(1,4) rmse_res_deg0{8,3,1,1}(1,4)  rmse_res_deg0{8,4,1,1}(1,4) rmse_res_deg0{8,5,1,1}(1,4) rmse_res_deg0{8,6,1,1}(1,4)];
% 
% mat1_100_deg = [rmse_res_deg1{8,1,1,1}(1,1) rmse_res_deg1{8,2,1,1}(1,1) rmse_res_deg1{8,3,1,1}(1,1)  rmse_res_deg1{8,4,1,1}(1,1) rmse_res_deg1{8,5,1,1}(1,1) rmse_res_deg1{8,6,1,1}(1,1); 
% rmse_res_deg1{8,1,1,1}(1,2) rmse_res_deg1{8,2,1,1}(1,2) rmse_res_deg1{8,3,1,1}(1,2)  rmse_res_deg1{8,4,1,1}(1,2) rmse_res_deg1{8,5,1,1}(1,2) rmse_res_deg1{8,6,1,1}(1,2);
% rmse_res_deg1{8,1,1,1}(1,3) rmse_res_deg1{8,2,1,1}(1,3) rmse_res_deg1{8,3,1,1}(1,3)  rmse_res_deg1{8,4,1,1}(1,3) rmse_res_deg1{8,5,1,1}(1,3) rmse_res_deg1{8,6,1,1}(1,3);
% rmse_res_deg1{8,1,1,1}(1,4) rmse_res_deg1{8,2,1,1}(1,4) rmse_res_deg1{8,3,1,1}(1,4)  rmse_res_deg1{8,4,1,1}(1,4) rmse_res_deg1{8,5,1,1}(1,4) rmse_res_deg1{8,6,1,1}(1,4)];
% 
% mat2_100_deg = [rmse_res_deg2{8,1,1,1}(1,1) rmse_res_deg2{8,2,1,1}(1,1) rmse_res_deg2{8,3,1,1}(1,1)  rmse_res_deg2{8,4,1,1}(1,1) rmse_res_deg2{8,5,1,1}(1,1) rmse_res_deg2{8,6,1,1}(1,1); 
% rmse_res_deg2{8,1,1,1}(1,2) rmse_res_deg2{8,2,1,1}(1,2) rmse_res_deg2{8,3,1,1}(1,2)  rmse_res_deg2{8,4,1,1}(1,2) rmse_res_deg2{8,5,1,1}(1,2) rmse_res_deg2{8,6,1,1}(1,2);
% rmse_res_deg2{8,1,1,1}(1,3) rmse_res_deg2{8,2,1,1}(1,3) rmse_res_deg2{8,3,1,1}(1,3)  rmse_res_deg2{8,4,1,1}(1,3) rmse_res_deg2{8,5,1,1}(1,3) rmse_res_deg2{8,6,1,1}(1,3);
% rmse_res_deg2{8,1,1,1}(1,4) rmse_res_deg2{8,2,1,1}(1,4) rmse_res_deg2{8,3,1,1}(1,4)  rmse_res_deg2{8,4,1,1}(1,4) rmse_res_deg2{8,5,1,1}(1,4) rmse_res_deg2{8,6,1,1}(1,4)];
% 
% mat0_250_deg = [rmse_res_deg0{8,1,1,2}(1,1) rmse_res_deg0{8,2,1,2}(1,1) rmse_res_deg0{8,3,1,2}(1,1)  rmse_res_deg0{8,4,1,2}(1,1) rmse_res_deg0{8,5,1,2}(1,1) rmse_res_deg0{8,6,1,2}(1,1); 
% rmse_res_deg0{8,1,1,2}(1,2) rmse_res_deg0{8,2,1,2}(1,2) rmse_res_deg0{8,3,1,2}(1,2)  rmse_res_deg0{8,4,1,2}(1,2) rmse_res_deg0{8,5,1,2}(1,2) rmse_res_deg0{8,6,1,2}(1,2);
% rmse_res_deg0{8,1,1,2}(1,3) rmse_res_deg0{8,2,1,2}(1,3) rmse_res_deg0{8,3,1,2}(1,3)  rmse_res_deg0{8,4,1,2}(1,3) rmse_res_deg0{8,5,1,2}(1,3) rmse_res_deg0{8,6,1,2}(1,3);
% rmse_res_deg0{8,1,1,2}(1,4) rmse_res_deg0{8,2,1,2}(1,4) rmse_res_deg0{8,3,1,2}(1,4)  rmse_res_deg0{8,4,1,2}(1,4) rmse_res_deg0{8,5,1,2}(1,4) rmse_res_deg0{8,6,1,2}(1,4)];
% 
% mat1_250_deg = [rmse_res_deg1{8,1,1,2}(1,1) rmse_res_deg1{8,2,1,2}(1,1) rmse_res_deg1{8,3,1,2}(1,1)  rmse_res_deg1{8,4,1,2}(1,1) rmse_res_deg1{8,5,1,2}(1,1) rmse_res_deg1{8,6,1,2}(1,1); 
% rmse_res_deg1{8,1,1,2}(1,2) rmse_res_deg1{8,2,1,2}(1,2) rmse_res_deg1{8,3,1,2}(1,2)  rmse_res_deg1{8,4,1,2}(1,2) rmse_res_deg1{8,5,1,2}(1,2) rmse_res_deg1{8,6,1,2}(1,2);
% rmse_res_deg1{8,1,1,2}(1,3) rmse_res_deg1{8,2,1,2}(1,3) rmse_res_deg1{8,3,1,2}(1,3)  rmse_res_deg1{8,4,1,2}(1,3) rmse_res_deg1{8,5,1,2}(1,3) rmse_res_deg1{8,6,1,2}(1,3);
% rmse_res_deg1{8,1,1,2}(1,4) rmse_res_deg1{8,2,1,2}(1,4) rmse_res_deg1{8,3,1,2}(1,4)  rmse_res_deg1{8,4,1,2}(1,4) rmse_res_deg1{8,5,1,2}(1,4) rmse_res_deg1{8,6,1,2}(1,4)];
% 
% mat2_250_deg = [rmse_res_deg2{8,1,1,2}(1,1) rmse_res_deg2{8,2,1,2}(1,1) rmse_res_deg2{8,3,1,2}(1,1)  rmse_res_deg2{8,4,1,2}(1,1) rmse_res_deg2{8,5,1,2}(1,1) rmse_res_deg2{8,6,1,2}(1,1); 
% rmse_res_deg2{8,1,1,2}(1,2) rmse_res_deg2{8,2,1,2}(1,2) rmse_res_deg2{8,3,1,2}(1,2)  rmse_res_deg2{8,4,1,2}(1,2) rmse_res_deg2{8,5,1,2}(1,2) rmse_res_deg2{8,6,1,2}(1,2);
% rmse_res_deg2{8,1,1,2}(1,3) rmse_res_deg2{8,2,1,2}(1,3) rmse_res_deg2{8,3,1,2}(1,3)  rmse_res_deg2{8,4,1,2}(1,3) rmse_res_deg2{8,5,1,2}(1,3) rmse_res_deg2{8,6,1,2}(1,3);
% rmse_res_deg2{8,1,1,2}(1,4) rmse_res_deg2{8,2,1,2}(1,4) rmse_res_deg2{8,3,1,2}(1,4)  rmse_res_deg2{8,4,1,2}(1,4) rmse_res_deg2{8,5,1,2}(1,4) rmse_res_deg2{8,6,1,2}(1,4)];
mat0_100_h = [rmse_res_h0{d,1,1,1}(1,1) rmse_res_h0{d,2,1,1}(1,1) rmse_res_h0{d,3,1,1}(1,1)  rmse_res_h0{d,4,1,1}(1,1) rmse_res_h0{d,5,1,1}(1,1) rmse_res_h0{d,6,1,1}(1,1); 
rmse_res_h0{d,1,1,1}(1,2) rmse_res_h0{d,2,1,1}(1,2) rmse_res_h0{d,3,1,1}(1,2)  rmse_res_h0{d,4,1,1}(1,2) rmse_res_h0{d,5,1,1}(1,2) rmse_res_h0{d,6,1,1}(1,2);
rmse_res_h0{d,1,1,1}(1,3) rmse_res_h0{d,2,1,1}(1,3) rmse_res_h0{d,3,1,1}(1,3)  rmse_res_h0{d,4,1,1}(1,3) rmse_res_h0{d,5,1,1}(1,3) rmse_res_h0{d,6,1,1}(1,3);
rmse_res_h0{d,1,1,1}(1,4) rmse_res_h0{d,2,1,1}(1,4) rmse_res_h0{d,3,1,1}(1,4)  rmse_res_h0{d,4,1,1}(1,4) rmse_res_h0{d,5,1,1}(1,4) rmse_res_h0{d,6,1,1}(1,4)];

mat1_100_h = [rmse_res_h1{d,1,1,1}(1,1) rmse_res_h1{d,2,1,1}(1,1) rmse_res_h1{d,3,1,1}(1,1)  rmse_res_h1{d,4,1,1}(1,1) rmse_res_h1{d,5,1,1}(1,1) rmse_res_h1{d,6,1,1}(1,1); 
rmse_res_h1{d,1,1,1}(1,2) rmse_res_h1{d,2,1,1}(1,2) rmse_res_h1{d,3,1,1}(1,2)  rmse_res_h1{d,4,1,1}(1,2) rmse_res_h1{d,5,1,1}(1,2) rmse_res_h1{d,6,1,1}(1,2);
rmse_res_h1{d,1,1,1}(1,3) rmse_res_h1{d,2,1,1}(1,3) rmse_res_h1{d,3,1,1}(1,3)  rmse_res_h1{d,4,1,1}(1,3) rmse_res_h1{d,5,1,1}(1,3) rmse_res_h1{d,6,1,1}(1,3);
rmse_res_h1{d,1,1,1}(1,4) rmse_res_h1{d,2,1,1}(1,4) rmse_res_h1{d,3,1,1}(1,4)  rmse_res_h1{d,4,1,1}(1,4) rmse_res_h1{d,5,1,1}(1,4) rmse_res_h1{d,6,1,1}(1,4)];

mat2_100_h = [rmse_res_h2{d,1,1,1}(1,1) rmse_res_h2{d,2,1,1}(1,1) rmse_res_h2{d,3,1,1}(1,1)  rmse_res_h2{d,4,1,1}(1,1) rmse_res_h2{d,5,1,1}(1,1) rmse_res_h2{d,6,1,1}(1,1); 
rmse_res_h2{d,1,1,1}(1,2) rmse_res_h2{d,2,1,1}(1,2) rmse_res_h2{d,3,1,1}(1,2)  rmse_res_h2{d,4,1,1}(1,2) rmse_res_h2{d,5,1,1}(1,2) rmse_res_h2{d,6,1,1}(1,2);
rmse_res_h2{d,1,1,1}(1,3) rmse_res_h2{d,2,1,1}(1,3) rmse_res_h2{d,3,1,1}(1,3)  rmse_res_h2{d,4,1,1}(1,3) rmse_res_h2{d,5,1,1}(1,3) rmse_res_h2{d,6,1,1}(1,3);
rmse_res_h2{d,1,1,1}(1,4) rmse_res_h2{d,2,1,1}(1,4) rmse_res_h2{d,3,1,1}(1,4)  rmse_res_h2{d,4,1,1}(1,4) rmse_res_h2{d,5,1,1}(1,4) rmse_res_h2{d,6,1,1}(1,4)];

mat0_250_h = [rmse_res_h0{d,1,1,2}(1,1) rmse_res_h0{d,2,1,2}(1,1) rmse_res_h0{d,3,1,2}(1,1)  rmse_res_h0{d,4,1,2}(1,1) rmse_res_h0{d,5,1,2}(1,1) rmse_res_h0{d,6,1,2}(1,1); 
rmse_res_h0{d,1,1,2}(1,2) rmse_res_h0{d,2,1,2}(1,2) rmse_res_h0{d,3,1,2}(1,2)  rmse_res_h0{d,4,1,2}(1,2) rmse_res_h0{d,5,1,2}(1,2) rmse_res_h0{d,6,1,2}(1,2);
rmse_res_h0{d,1,1,2}(1,3) rmse_res_h0{d,2,1,2}(1,3) rmse_res_h0{d,3,1,2}(1,3)  rmse_res_h0{d,4,1,2}(1,3) rmse_res_h0{d,5,1,2}(1,3) rmse_res_h0{d,6,1,2}(1,3);
rmse_res_h0{d,1,1,2}(1,4) rmse_res_h0{d,2,1,2}(1,4) rmse_res_h0{d,3,1,2}(1,4)  rmse_res_h0{d,4,1,2}(1,4) rmse_res_h0{d,5,1,2}(1,4) rmse_res_h0{d,6,1,2}(1,4)];

mat1_250_h = [rmse_res_h1{d,1,1,2}(1,1) rmse_res_h1{d,2,1,2}(1,1) rmse_res_h1{d,3,1,2}(1,1)  rmse_res_h1{d,4,1,2}(1,1) rmse_res_h1{d,5,1,2}(1,1) rmse_res_h1{d,6,1,2}(1,1); 
rmse_res_h1{d,1,1,2}(1,2) rmse_res_h1{d,2,1,2}(1,2) rmse_res_h1{d,3,1,2}(1,2)  rmse_res_h1{d,4,1,2}(1,2) rmse_res_h1{d,5,1,2}(1,2) rmse_res_h1{d,6,1,2}(1,2);
rmse_res_h1{d,1,1,2}(1,3) rmse_res_h1{d,2,1,2}(1,3) rmse_res_h1{d,3,1,2}(1,3)  rmse_res_h1{d,4,1,2}(1,3) rmse_res_h1{d,5,1,2}(1,3) rmse_res_h1{d,6,1,2}(1,3);
rmse_res_h1{d,1,1,2}(1,4) rmse_res_h1{d,2,1,2}(1,4) rmse_res_h1{d,3,1,2}(1,4)  rmse_res_h1{d,4,1,2}(1,4) rmse_res_h1{d,5,1,2}(1,4) rmse_res_h1{d,6,1,2}(1,4)];

mat2_250_h = [rmse_res_h2{d,1,1,2}(1,1) rmse_res_h2{d,2,1,2}(1,1) rmse_res_h2{d,3,1,2}(1,1)  rmse_res_h2{d,4,1,2}(1,1) rmse_res_h2{d,5,1,2}(1,1) rmse_res_h2{d,6,1,2}(1,1); 
rmse_res_h2{d,1,1,2}(1,2) rmse_res_h2{d,2,1,2}(1,2) rmse_res_h2{d,3,1,2}(1,2)  rmse_res_h2{d,4,1,2}(1,2) rmse_res_h2{d,5,1,2}(1,2) rmse_res_h2{d,6,1,2}(1,2);
rmse_res_h2{d,1,1,2}(1,3) rmse_res_h2{d,2,1,2}(1,3) rmse_res_h2{d,3,1,2}(1,3)  rmse_res_h2{d,4,1,2}(1,3) rmse_res_h2{d,5,1,2}(1,3) rmse_res_h2{d,6,1,2}(1,3);
rmse_res_h2{d,1,1,2}(1,4) rmse_res_h2{d,2,1,2}(1,4) rmse_res_h2{d,3,1,2}(1,4)  rmse_res_h2{d,4,1,2}(1,4) rmse_res_h2{d,5,1,2}(1,4) rmse_res_h2{d,6,1,2}(1,4)];

%%deg

mat0_100_deg = [rmse_res_deg0{d,1,1,1}(1,1) rmse_res_deg0{d,2,1,1}(1,1) rmse_res_deg0{d,3,1,1}(1,1)  rmse_res_deg0{d,4,1,1}(1,1) rmse_res_deg0{d,5,1,1}(1,1) rmse_res_deg0{d,6,1,1}(1,1); 
rmse_res_deg0{d,1,1,1}(1,2) rmse_res_deg0{d,2,1,1}(1,2) rmse_res_deg0{d,3,1,1}(1,2)  rmse_res_deg0{d,4,1,1}(1,2) rmse_res_deg0{d,5,1,1}(1,2) rmse_res_deg0{d,6,1,1}(1,2);
rmse_res_deg0{d,1,1,1}(1,3) rmse_res_deg0{d,2,1,1}(1,3) rmse_res_deg0{d,3,1,1}(1,3)  rmse_res_deg0{d,4,1,1}(1,3) rmse_res_deg0{d,5,1,1}(1,3) rmse_res_deg0{d,6,1,1}(1,3);
rmse_res_deg0{d,1,1,1}(1,4) rmse_res_deg0{d,2,1,1}(1,4) rmse_res_deg0{d,3,1,1}(1,4)  rmse_res_deg0{d,4,1,1}(1,4) rmse_res_deg0{d,5,1,1}(1,4) rmse_res_deg0{d,6,1,1}(1,4)];

mat1_100_deg = [rmse_res_deg1{d,1,1,1}(1,1) rmse_res_deg1{d,2,1,1}(1,1) rmse_res_deg1{d,3,1,1}(1,1)  rmse_res_deg1{d,4,1,1}(1,1) rmse_res_deg1{d,5,1,1}(1,1) rmse_res_deg1{d,6,1,1}(1,1); 
rmse_res_deg1{d,1,1,1}(1,2) rmse_res_deg1{d,2,1,1}(1,2) rmse_res_deg1{d,3,1,1}(1,2)  rmse_res_deg1{d,4,1,1}(1,2) rmse_res_deg1{d,5,1,1}(1,2) rmse_res_deg1{d,6,1,1}(1,2);
rmse_res_deg1{d,1,1,1}(1,3) rmse_res_deg1{d,2,1,1}(1,3) rmse_res_deg1{d,3,1,1}(1,3)  rmse_res_deg1{d,4,1,1}(1,3) rmse_res_deg1{d,5,1,1}(1,3) rmse_res_deg1{d,6,1,1}(1,3);
rmse_res_deg1{d,1,1,1}(1,4) rmse_res_deg1{d,2,1,1}(1,4) rmse_res_deg1{d,3,1,1}(1,4)  rmse_res_deg1{d,4,1,1}(1,4) rmse_res_deg1{d,5,1,1}(1,4) rmse_res_deg1{d,6,1,1}(1,4)];

mat2_100_deg = [rmse_res_deg2{d,1,1,1}(1,1) rmse_res_deg2{d,2,1,1}(1,1) rmse_res_deg2{d,3,1,1}(1,1)  rmse_res_deg2{d,4,1,1}(1,1) rmse_res_deg2{d,5,1,1}(1,1) rmse_res_deg2{d,6,1,1}(1,1); 
rmse_res_deg2{d,1,1,1}(1,2) rmse_res_deg2{d,2,1,1}(1,2) rmse_res_deg2{d,3,1,1}(1,2)  rmse_res_deg2{d,4,1,1}(1,2) rmse_res_deg2{d,5,1,1}(1,2) rmse_res_deg2{d,6,1,1}(1,2);
rmse_res_deg2{d,1,1,1}(1,3) rmse_res_deg2{d,2,1,1}(1,3) rmse_res_deg2{d,3,1,1}(1,3)  rmse_res_deg2{d,4,1,1}(1,3) rmse_res_deg2{d,5,1,1}(1,3) rmse_res_deg2{d,6,1,1}(1,3);
rmse_res_deg2{d,1,1,1}(1,4) rmse_res_deg2{d,2,1,1}(1,4) rmse_res_deg2{d,3,1,1}(1,4)  rmse_res_deg2{d,4,1,1}(1,4) rmse_res_deg2{d,5,1,1}(1,4) rmse_res_deg2{d,6,1,1}(1,4)];

mat0_250_deg = [rmse_res_deg0{d,1,1,2}(1,1) rmse_res_deg0{d,2,1,2}(1,1) rmse_res_deg0{d,3,1,2}(1,1)  rmse_res_deg0{d,4,1,2}(1,1) rmse_res_deg0{d,5,1,2}(1,1) rmse_res_deg0{d,6,1,2}(1,1); 
rmse_res_deg0{d,1,1,2}(1,2) rmse_res_deg0{d,2,1,2}(1,2) rmse_res_deg0{d,3,1,2}(1,2)  rmse_res_deg0{d,4,1,2}(1,2) rmse_res_deg0{d,5,1,2}(1,2) rmse_res_deg0{d,6,1,2}(1,2);
rmse_res_deg0{d,1,1,2}(1,3) rmse_res_deg0{d,2,1,2}(1,3) rmse_res_deg0{d,3,1,2}(1,3)  rmse_res_deg0{d,4,1,2}(1,3) rmse_res_deg0{d,5,1,2}(1,3) rmse_res_deg0{d,6,1,2}(1,3);
rmse_res_deg0{d,1,1,2}(1,4) rmse_res_deg0{d,2,1,2}(1,4) rmse_res_deg0{d,3,1,2}(1,4)  rmse_res_deg0{d,4,1,2}(1,4) rmse_res_deg0{d,5,1,2}(1,4) rmse_res_deg0{d,6,1,2}(1,4)];

mat1_250_deg = [rmse_res_deg1{d,1,1,2}(1,1) rmse_res_deg1{d,2,1,2}(1,1) rmse_res_deg1{d,3,1,2}(1,1)  rmse_res_deg1{d,4,1,2}(1,1) rmse_res_deg1{d,5,1,2}(1,1) rmse_res_deg1{d,6,1,2}(1,1); 
rmse_res_deg1{d,1,1,2}(1,2) rmse_res_deg1{d,2,1,2}(1,2) rmse_res_deg1{d,3,1,2}(1,2)  rmse_res_deg1{d,4,1,2}(1,2) rmse_res_deg1{d,5,1,2}(1,2) rmse_res_deg1{d,6,1,2}(1,2);
rmse_res_deg1{d,1,1,2}(1,3) rmse_res_deg1{d,2,1,2}(1,3) rmse_res_deg1{d,3,1,2}(1,3)  rmse_res_deg1{d,4,1,2}(1,3) rmse_res_deg1{d,5,1,2}(1,3) rmse_res_deg1{d,6,1,2}(1,3);
rmse_res_deg1{d,1,1,2}(1,4) rmse_res_deg1{d,2,1,2}(1,4) rmse_res_deg1{d,3,1,2}(1,4)  rmse_res_deg1{d,4,1,2}(1,4) rmse_res_deg1{d,5,1,2}(1,4) rmse_res_deg1{d,6,1,2}(1,4)];

mat2_250_deg = [rmse_res_deg2{d,1,1,2}(1,1) rmse_res_deg2{d,2,1,2}(1,1) rmse_res_deg2{d,3,1,2}(1,1)  rmse_res_deg2{d,4,1,2}(1,1) rmse_res_deg2{d,5,1,2}(1,1) rmse_res_deg2{d,6,1,2}(1,1); 
rmse_res_deg2{d,1,1,2}(1,2) rmse_res_deg2{d,2,1,2}(1,2) rmse_res_deg2{d,3,1,2}(1,2)  rmse_res_deg2{d,4,1,2}(1,2) rmse_res_deg2{d,5,1,2}(1,2) rmse_res_deg2{d,6,1,2}(1,2);
rmse_res_deg2{d,1,1,2}(1,3) rmse_res_deg2{d,2,1,2}(1,3) rmse_res_deg2{d,3,1,2}(1,3)  rmse_res_deg2{d,4,1,2}(1,3) rmse_res_deg2{d,5,1,2}(1,3) rmse_res_deg2{d,6,1,2}(1,3);
rmse_res_deg2{d,1,1,2}(1,4) rmse_res_deg2{d,2,1,2}(1,4) rmse_res_deg2{d,3,1,2}(1,4)  rmse_res_deg2{d,4,1,2}(1,4) rmse_res_deg2{d,5,1,2}(1,4) rmse_res_deg2{d,6,1,2}(1,4)];

mat0_100_h;
filename = ['CV_dense_design' num2str(d) '.tex'];

FID = fopen(filename, 'w');
fprintf(FID, '\\begin{table}\\caption{\\footnotesize {\\bf Cross-Validation results}: Parameter values across %.0f Monte Carlo replications for dense network design %.0f} \n',B,d);
fprintf(FID,'\\begin{threeparttable} \n');
fprintf(FID, '\\centering \\footnotesize\n');
fprintf(FID, '\\scalebox{.9}{\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\\toprule \n');

%% b1

fprintf(FID,'&\\cellcolor{yellow}$N$&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}&\\multicolumn{6}{|c|}{\\cellcolor{yellow}$%.0f$}\\\\\\hline \n',100,250);
fprintf(FID,'&&\\multicolumn{6}{c}{$K_N$}&\\multicolumn{6}{c}{$K_N$}\\\\\\hline \n');

fprintf(FID,'&$\\beta_0-\\hat{\\beta_0}$&$3$&$4$&$5$&$6$&$7$& $8$ &$3$&$4$&$5$&$6$&$7$& $8$\\\\\\midrule \n');
fprintf(FID,'\\multicolumn{14}{c}{\\bf Control function: $\\widehat{h}(a_i)$}\\\\	\\midrule \n');
%%h(a)=exp(a)
fprintf(FID,'\\multirow{4}{*}{\\rotatebox[origin=c]{90}{$\\exp(a_i)$}}&mean&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat0_100_h(1,1),mat0_100_h(1,2),mat0_100_h(1,3),mat0_100_h(1,4),mat0_100_h(1,5),mat0_100_h(1,6),mat0_250_h(1,1),mat0_250_h(1,2),mat0_250_h(1,3),mat0_250_h(1,4),mat0_250_h(1,5),mat0_250_h(1,6));
fprintf(FID,'&median&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat0_100_h(2,1),mat0_100_h(2,2),mat0_100_h(2,3),mat0_100_h(2,4),mat0_100_h(2,5),mat0_100_h(2,6),mat0_250_h(2,1),mat0_250_h(2,2),mat0_250_h(2,3),mat0_250_h(2,4),mat0_250_h(2,5),mat0_250_h(2,6));
fprintf(FID,'&std&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat0_100_h(3,1),mat0_100_h(3,2),mat0_100_h(3,3),mat0_100_h(3,4),mat0_100_h(3,5),mat0_100_h(3,6),mat0_250_h(3,1),mat0_250_h(3,2),mat0_250_h(3,3),mat0_250_h(3,4),mat0_250_h(3,5),mat0_250_h(3,6));
fprintf(FID,'&iqr&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\\\midrule \n',mat0_100_h(4,1),mat0_100_h(4,2),mat0_100_h(4,3),mat0_100_h(4,4),mat0_100_h(4,5),mat0_100_h(4,6),mat0_250_h(4,1),mat0_250_h(4,2),mat0_250_h(4,3),mat0_250_h(4,4),mat0_250_h(4,5),mat0_250_h(4,6));

%%h(a)=cos(a)
fprintf(FID,'\\multirow{4}{*}{\\rotatebox[origin=c]{90}{$\\cos(a_i)$}}&mean&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat1_100_h(1,1),mat1_100_h(1,2),mat1_100_h(1,3),mat1_100_h(1,4),mat1_100_h(1,5),mat1_100_h(1,6),mat1_250_h(1,1),mat1_250_h(1,2),mat1_250_h(1,3),mat1_250_h(1,4),mat1_250_h(1,5),mat1_250_h(1,6));
fprintf(FID,'&median&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat1_100_h(2,1),mat1_100_h(2,2),mat1_100_h(2,3),mat1_100_h(2,4),mat1_100_h(2,5),mat1_100_h(2,6),mat1_250_h(2,1),mat1_250_h(2,2),mat1_250_h(2,3),mat1_250_h(2,4),mat1_250_h(2,5),mat1_250_h(2,6));
fprintf(FID,'&std&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat1_100_h(3,1),mat1_100_h(3,2),mat1_100_h(3,3),mat1_100_h(3,4),mat1_100_h(3,5),mat1_100_h(3,6),mat1_250_h(3,1),mat1_250_h(3,2),mat1_250_h(3,3),mat1_250_h(3,4),mat1_250_h(3,5),mat1_250_h(3,6));
fprintf(FID,'&iqr&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\\\midrule \n',mat1_100_h(4,1),mat1_100_h(4,2),mat1_100_h(4,3),mat1_100_h(4,4),mat1_100_h(4,5),mat1_100_h(4,6),mat1_250_h(4,1),mat1_250_h(4,2),mat1_250_h(4,3),mat1_250_h(4,4),mat1_250_h(4,5),mat1_250_h(4,6));
%%h(a)=sin(a)
fprintf(FID,'\\multirow{4}{*}{\\rotatebox[origin=c]{90}{$\\sin(a_i)$}}&mean&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat2_100_h(1,1),mat2_100_h(1,2),mat2_100_h(1,3),mat2_100_h(1,4),mat2_100_h(1,5),mat2_100_h(1,6),mat2_250_h(1,1),mat2_250_h(1,2),mat2_250_h(1,3),mat2_250_h(1,4),mat2_250_h(1,5),mat2_250_h(1,6));
fprintf(FID,'&median&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat2_100_h(2,1),mat2_100_h(2,2),mat2_100_h(2,3),mat2_100_h(2,4),mat2_100_h(2,5),mat2_100_h(2,6),mat2_250_h(2,1),mat2_250_h(2,2),mat2_250_h(2,3),mat2_250_h(2,4),mat2_250_h(2,5),mat2_250_h(2,6));
fprintf(FID,'&std&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat2_100_h(3,1),mat2_100_h(3,2),mat2_100_h(3,3),mat2_100_h(3,4),mat2_100_h(3,5),mat2_100_h(3,6),mat2_250_h(3,1),mat2_250_h(3,2),mat2_250_h(3,3),mat2_250_h(3,4),mat2_250_h(3,5),mat2_250_h(3,6));
fprintf(FID,'&iqr&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\\\bottomrule \n',mat2_100_h(4,1),mat2_100_h(4,2),mat2_100_h(4,3),mat2_100_h(4,4),mat2_100_h(4,5),mat2_100_h(4,6),mat2_250_h(4,1),mat2_250_h(4,2),mat2_250_h(4,3),mat2_250_h(4,4),mat2_250_h(4,5),mat2_250_h(4,6));
fprintf(FID,'\\toprule \n');
fprintf(FID,'\\multicolumn{14}{c}{\\bf Control function: $\\widehat{h}(\\widehat{deg_i},a_i)$}\\\\	\\midrule \n');

%%h(a)=exp(a)
fprintf(FID,'\\multirow{4}{*}{\\rotatebox[origin=c]{90}{$\\exp(a_i)$}}&mean&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat0_100_deg(1,1),mat0_100_deg(1,2),mat0_100_deg(1,3),mat0_100_deg(1,4),mat0_100_deg(1,5),mat0_100_deg(1,6),mat0_250_deg(1,1),mat0_250_deg(1,2),mat0_250_deg(1,3),mat0_250_deg(1,4),mat0_250_deg(1,5),mat0_250_deg(1,6));
fprintf(FID,'&median&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat0_100_deg(2,1),mat0_100_deg(2,2),mat0_100_deg(2,3),mat0_100_deg(2,4),mat0_100_deg(2,5),mat0_100_deg(2,6),mat0_250_deg(2,1),mat0_250_deg(2,2),mat0_250_deg(2,3),mat0_250_deg(2,4),mat0_250_deg(2,5),mat0_250_deg(2,6));
fprintf(FID,'&std&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat0_100_deg(3,1),mat0_100_deg(3,2),mat0_100_deg(3,3),mat0_100_deg(3,4),mat0_100_deg(3,5),mat0_100_deg(3,6),mat0_250_deg(3,1),mat0_250_deg(3,2),mat0_250_deg(3,3),mat0_250_deg(3,4),mat0_250_deg(3,5),mat0_250_deg(3,6));
fprintf(FID,'&iqr&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\\\midrule \n',mat0_100_deg(4,1),mat0_100_deg(4,2),mat0_100_deg(4,3),mat0_100_deg(4,4),mat0_100_deg(4,5),mat0_100_deg(4,6),mat0_250_deg(4,1),mat0_250_deg(4,2),mat0_250_deg(4,3),mat0_250_deg(4,4),mat0_250_deg(4,5),mat0_250_deg(4,6));

%%h(a)=cos(a)
fprintf(FID,'\\multirow{4}{*}{\\rotatebox[origin=c]{90}{$\\cos(a_i)$}}&mean&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat1_100_deg(1,1),mat1_100_deg(1,2),mat1_100_deg(1,3),mat1_100_deg(1,4),mat1_100_deg(1,5),mat1_100_deg(1,6),mat1_250_deg(1,1),mat1_250_deg(1,2),mat1_250_deg(1,3),mat1_250_deg(1,4),mat1_250_deg(1,5),mat1_250_deg(1,6));
fprintf(FID,'&median&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat1_100_deg(2,1),mat1_100_deg(2,2),mat1_100_deg(2,3),mat1_100_deg(2,4),mat1_100_deg(2,5),mat1_100_deg(2,6),mat1_250_deg(2,1),mat1_250_deg(2,2),mat1_250_deg(2,3),mat1_250_deg(2,4),mat1_250_deg(2,5),mat1_250_deg(2,6));
fprintf(FID,'&std&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat1_100_deg(3,1),mat1_100_deg(3,2),mat1_100_deg(3,3),mat1_100_deg(3,4),mat1_100_deg(3,5),mat1_100_deg(3,6),mat1_250_deg(3,1),mat1_250_deg(3,2),mat1_250_deg(3,3),mat1_250_deg(3,4),mat1_250_deg(3,5),mat1_250_deg(3,6));
fprintf(FID,'&iqr&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\\\midrule \n',mat1_100_deg(4,1),mat1_100_deg(4,2),mat1_100_deg(4,3),mat1_100_deg(4,4),mat1_100_deg(4,5),mat1_100_deg(4,6),mat1_250_deg(4,1),mat1_250_deg(4,2),mat1_250_deg(4,3),mat1_250_deg(4,4),mat1_250_deg(4,5),mat1_250_deg(4,6));
%%h(a)=sin(a)
fprintf(FID,'\\multirow{4}{*}{\\rotatebox[origin=c]{90}{$\\sin(a_i)$}}&mean&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat2_100_deg(1,1),mat2_100_deg(1,2),mat2_100_deg(1,3),mat2_100_deg(1,4),mat2_100_deg(1,5),mat2_100_deg(1,6),mat2_250_deg(1,1),mat2_250_deg(1,2),mat2_250_deg(1,3),mat2_250_deg(1,4),mat2_250_deg(1,5),mat2_250_deg(1,6));
fprintf(FID,'&median&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat2_100_deg(2,1),mat2_100_deg(2,2),mat2_100_deg(2,3),mat2_100_deg(2,4),mat2_100_deg(2,5),mat2_100_deg(2,6),mat2_250_deg(2,1),mat2_250_deg(2,2),mat2_250_deg(2,3),mat2_250_deg(2,4),mat2_250_deg(2,5),mat2_250_deg(2,6));
fprintf(FID,'&std&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\ \n',mat2_100_deg(3,1),mat2_100_deg(3,2),mat2_100_deg(3,3),mat2_100_deg(3,4),mat2_100_deg(3,5),mat2_100_deg(3,6),mat2_250_deg(3,1),mat2_250_deg(3,2),mat2_250_deg(3,3),mat2_250_deg(3,4),mat2_250_deg(3,5),mat2_250_deg(3,6));
fprintf(FID,'&iqr&%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f &%.3f \\\\\\bottomrule \n',mat2_100_deg(4,1),mat2_100_deg(4,2),mat2_100_deg(4,3),mat2_100_deg(4,4),mat2_100_deg(4,5),mat2_100_deg(4,6),mat2_250_deg(4,1),mat2_250_deg(4,2),mat2_250_deg(4,3),mat2_250_deg(4,4),mat2_250_deg(4,5),mat2_250_deg(4,6));

fprintf(FID,'\\end{tabular}} \n');
%tablenotes
fprintf(FID,'\\begin{tablenotes}\\tiny \n');
fprintf(FID,'\\item The statistics are based on conventional leave one out cross-validation. \n');
fprintf(FID,'  \\end{tablenotes} \n');
fprintf(FID,'\\end{threeparttable} \n');
fprintf(FID,'\\end{table} \n');
fclose(FID);
% beta
end
