function G = gram_norm_matrix2(X1, X2, kernel)
% GRAM_NORM_MATRIX2 Computes the normalized Gram matrix between two sample sets
%
% SYNOPSIS: G = gram_matrix2(X1, X2, kernel)
%
% INPUT:
% - X1, X2: matrices containing one sample feature vector per row,
%           they must have the same number of columns
% - kernel: a function that computes the scalar product of two vectors
%           in feature space (takes row vectors)
%
% OUTPUT:
% - G: the matrix of products of sample vectors in feature space, via the
%      kernel function, of unit norm
%
% REMARKS:
% The Gram matrix is usually ill-conditioned, so avoid using its inverse directly

num_samples1 = size(X1, 1);
num_samples2 = size(X2, 1);

% matrix preallocation
G = zeros(num_samples1, num_samples2);

for i = 1:num_samples1
    for j = 1:num_samples2
        G(i,j) = kernel(X1(i,:), X2(j,:)) / (sqrt(kernel(X1(i,:), X1(i,:))) * sqrt(kernel(X2(j,:), X2(j,:))));
    end
end

end