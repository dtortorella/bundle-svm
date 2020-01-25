function [Q,R,P,sv] = online_gram_qr(G,tol)
%ONLINE_QR Computes a rank-k QR factorization of a Gram matrix estimating k
% using one sample at a time and repeated qr factorization,
% selecting k linear independent row/cols of G.
% 
% SYNOPSIS: [Q,R,P,sv] = online_qr(G,tol)
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
% - P: permutation matrix, resulting from QR factorization
% - sv: vector of indices of the selected vectors, corresponding to rows/cols of G
%
% REMARKS:
% See: matlab qr

A = G(1,1);
sv = [1];
k=1;
for i = 2:size(G,1)
    a = G(i,1:end-1);
    a = a(sv);
    b = G(i,i);
    
    [Q,R,P] = qr([A a'; a b]);
    
    if sum(abs(diag(R)) > tol) == k+1
        %we added orthogonal information and use this sample
        A = [A a'; a b];
        k = k+1;
        sv = [sv i];
    else
        [~,idx] = max(P(end,:)); %should be the index of the <tol sample
        if idx ~= i 
            sv(sv==idx) = i; %replace that sample with lastone. Should improve orthogonality
            
        end
    end
    
end
end

