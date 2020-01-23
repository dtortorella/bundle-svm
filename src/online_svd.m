function [U,S,sv] = online_svd(G,tol)
%ONLINE_SVD computes rank-m SVD of a Gram matrix estimating m 
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


A = G(1,1);
sv = [1];
k=1;

for i = 2:size(G,1)
    a = G(i,1:end-1);
    a = a(sv);
    b = G(i,i);
    
    [U,S,V] = svd([A a'; a b]);
    
    if sum(abs(diag(S)) > tol) == k+1
        %we added orthogonal information and use this sample
        A = [A a'; a b];
        k = k+1;
        sv = [sv i];
    end
    
end
end



