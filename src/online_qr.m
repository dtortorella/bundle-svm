function [Q,R,p,sv] = online_qr(G, tol)
%ONLINE_QR Computes a rank-k QR factorization of a Gram matrix estimating k
% using one sample at a time and repeated qr factorization,
% selecting k linear independent row/cols of G.
% 
% SYNOPSIS: [Q,R,p,sv] = online_qr(G,tol)
%
% INPUT:
% - G: the matrix of products of sample vectors in feature space, via the
%      kernel function (Gram matrix)
% - tol: threshold. diagonal values of R with absolute value lower than tol
%        will be considered 0
%
% OUTPUT:
% - Q: orthogonal matrix from the QR factorization such that G(sv) = Q*R*P'
% - R: upper triangular from the QR factorization 
% - p: permutation vector, resulting from QR factorization
% - sv: vector of indices of the selected vectors, corresponding to rows/cols of G
%
% REMARKS:
% See: matlab qr

[Q,R,p] = qr(G(1,1), 0);
sv = [1];

for i = 2:size(G, 1)
    % new factorization
    svi = [sv i];
    [Qi,Ri,pi] = qr(G(svi,svi), 0);
    
    % samples to exclude
    exclude = abs(diag(Ri)) < tol;
    
    if sum(exclude) == 0
        % all samples above threshold
        Q = Qi; R = Ri; p = pi;
        sv = svi;
    else
        % new sample is below threshold
        Q = Qi(~exclude,~exclude);
        R = Ri(~exclude,~exclude);
        p = pi(~exclude);
        sv = svi(~exclude);
    end
end

end
