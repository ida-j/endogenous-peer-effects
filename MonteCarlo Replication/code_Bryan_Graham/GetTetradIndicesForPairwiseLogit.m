function [dyad_pair_indices, ij_in_tetrad_indicators] = GetTetradIndicesForPairwiseLogit(N)


% SUMMARY: This function does the combinatoric calculations needed to create all the
%          various dyad & tetrad indices appearing in the pairwise logit criterion
%          function. The different index sets produced by this function speed up
%          computation. These index sets are required inputs into the
%          functions OrganizeDataForPairwiseLogit() and
%          betaSNM_PairwiseLogit()

% INPUTS:   N                       :   number of agents in the network
% OUTPUTS:  dyad_pair_indices       :   dyad indices for each of the
%                                       six dyads appearing in each tetrad 
%           ij_in_tetrad_indicators :   matrix indicating in which tetrads each of        
%                                       the 0.5N(N-1) dyads appear

%-------------------------------------------------------------------------%
%- Calculate Dyad & Tetrad Indices                                       -%
%-------------------------------------------------------------------------%

% Number of dyads in a network with N agents
n = N*(N-1)/2;

tic
disp('Enumerate all dyads in the network');
dyad_indices = nchoosek((1:N),2);
toc

tic
disp('Enumerate all tetrads in the network');
tetrad_indices = nchoosek((1:N),4);
toc

% Extract individual level individuals associated with each tetrad
i = tetrad_indices(:,1);
j = tetrad_indices(:,2);
k = tetrad_indices(:,3);
l = tetrad_indices(:,4);

% Enumerate the six unqiue dyads appearing in each tetrad
ij = sub2ind([N N], i,j);
ik = sub2ind([N N], i,k);
il = sub2ind([N N], i,l);
jk = sub2ind([N N], j,k);
jl = sub2ind([N N], j,l);
kl = sub2ind([N N], k,l);
dyad_pair_indices = [ij ik il jk jl kl];
clear i j k l ij ik il jk jl kl;

%-------------------------------------------------------------------------%
%- Enumerate tetrad membership for each dyad                             -%
%-------------------------------------------------------------------------%

% Each dyad appears appears in (N-2) choose 2 difference tetrads. These next
% few lines of code record in which of the N choose 4 tetrads each of the N
% choose 2 dyads appears.

% Initialize matrix which keeps track of in which of the N choose 4 tetrads
% each of the N choose 2 dyads appear

ij_in_tetrad_indicators = sparse(n,0.5*(N-2)*(N-3));

tic
disp('Enumerate tetrad membership indices for each dyad in the network');
parfor d = 1:n
        % find agents in dyad d
        i = dyad_indices(d,1);
        j = dyad_indices(d,2);
        
        % find indices for tetrads which contain both i and j
        ij_in_tetrad_indicators(d,:) = find(any(tetrad_indices==i ,2) .* any(tetrad_indices==j ,2))';
        
end
toc

end

