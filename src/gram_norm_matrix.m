function G = gram_norm_matrix(X, kernel)
%GRAM_MATRIX Computes the normalized Gram matrix of a sample set
%
% SYNOPSIS: G = gram_norm_matrix(X, kernel)
%
% INPUT:
% - X: a matrix containing one sample feature vector per row
% - kernel: a function that computes the scalar product of two vectors
%           in feature space (takes row vectors)
%
% OUTPUT:
% - G: the matrix of products of sample vectors in feature space, via the
%      kernel function, of unit norm
%
% REMARKS:
% The Gram matrix is usually ill-conditioned, so avoid using its inverse directly

num_samples = size(X, 1);

% matrix preallocation
G = zeros(num_samples);

for i = 1:num_samples
    j = 1;
    while j < i
        G(i,j) = kernel(X(i,:), X(j,:)) / (sqrt(kernel(X(i,:), X(i,:))) * sqrt(kernel(X(j,:), X(j,:))));
        G(j,i) = G(i,j);
        j = j + 1;
    end
    G(i,i) = kernel(X(i,:), X(i,:)) / kernel(X(i,:), X(i,:));
end

end