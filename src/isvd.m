function [U,S,sv] = isvd(G, tol)
%ISVD computes rank-m SVD of a Gram matrix estimating m 
% using one sample at a time and repeated svd decomposition, selecting m
% indipendent row/cols of the matrix G.
% 
% SYNOPSIS: [U,S,sv] = online_svd(G,tol)
%
% INPUT:
% - G: the matrix of products of sample vectors in feature space, via the
%      kernel function (Gram matrix)
% - tol: threshold. Singolar values with absolute value lower than tol will
%        be considered 0
%
% OUTPUT:
% - U: orthogonal matrix from svd decomposition such that G(sv) = U*S*U'
% - S: diagonal matrix of singolar values from decomposition
% - sv: vector of indices of the selected vectors, corresponding to rows/cols of G
%
% REMARKS:
% See: matlab svd


[U,S,~] = svd(G(1,1));
sv = [1];

for i = 2:size(G, 1)
    % new decomposition
    svi = [sv i];
    [Ui,Si,~] = svd(G(svi,svi));
    
    if sum(abs(diag(S)) < tol) == 0
        % all samples above threshold
        U = Ui; S = Si;
        sv = svi;
    end
end

end
