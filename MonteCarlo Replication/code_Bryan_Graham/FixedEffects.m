function [A_i1, NumIter, NoSol] = FixedEffects(A_i0,D,W,beta,tol,MaxIter)

% SUMMARY:          This function computes the concentrated MLEs of the N individual-specific
%                   heterogeneity parameters which enter the link
%                   probability in the augmented beta model of network formation. The
%                   procedure uses modification of the fixed point iteration method introduced
%                   in Chatterjee, Diaconis and Sly (2011, Annals of
%                   Applied Probability) described in Graham (2015)

% INPUTS:           A_i0    : N x 1 vector of starting values of A_i 
%                   D       : N X N adjacency matrix for the network
%                   W       : 0.5*N(N-1) X K matrix of dyad-level covariates 
%                   beta    : K x 1 vector of parameters on W
%                   tol     : Convergence criterion
%                   MaxIter : Maximum number of fixed point iterations allowed

% OUTPUTS:
%                   A_i1    : Next iteration of A_i vector (N x 1)
%                   NumIter : Number of fixed point iterations computed
%                   NoSol   : equals 1 if no fixed point, 0 otherwise

% CALLED BY                 : betaSNM_JointFixedEffects(),
%                             betaSNM_JointFixedEffects_log()
% FUNCTIONS CALLED          : FixedEffects_criterion()

sol             = 0;
NumIter         = 0;

while sol==0
    [A_i1] = FixedEffects_criterion(A_i0,D,W,beta);      % Iterate contraction
    if norm(A_i1-A_i0,'fro')<=tol                        % Check for convergence
        A_i0 = A_i1;                                     % If convergence, then update and exit   
        sol = 1;
    else                                                 % Else update starting values
        if NumIter<MaxIter                               % unless MaxIter has been reached
            A_i0 = A_i1;
            NumIter = NumIter + 1;
        else
            A_i0 = A_i1;                                 % If MaxIter reached, then update and exit 
            sol = 1;
        end
    end
    
end

% Check to see if fixed point exists
if sum(isnan(A_i1))>0
    NoSol = 1;              % If some A_i estimates are NaN, then there is no fixed point
else
    NoSol = 0;              % If all A_i are real valued, then assume there is a fixed point    
end

end

