function o = eval_orthonorm(X, kernel, varargin)
%EVAL_ORTHONORM Evaluates othonormality of a basis in kernel space
% Average pairwise cosines between vectors in kernel space is the metric.
%
% SYNOPSIS: o = eval_orthonorm(X, kernel)
%           o = eval_orthonorm(X, G)
%           o = eval_orthonorm(X, kernel, 'normalize')
%
% INPUT:
% - X: a matrix containing one sample feature vector per row
% - kernel: a function that computes the scalar product of two vectors
%           in feature space (takes row vectors)
% - G: normalized Gram matrix; if not empty, X are indices of this matrix
%
% OUTPUT:
% - o: an orthonormality metric, lower is better
%
% REMARKS:
% Consider normalizing this by the number of X vectors, for a proper
% comparison with other basis of different vectors.

% normalized Gram matrix

if isa(kernel,'function_handle')
    G = gram_norm_matrix(X, kernel);
else
    G = kernel;
    G = G(X,X);
end

% magnitude of off-diagonal cosines
o = sum(abs(eye(size(G)) - G), 'all');

% normalize by number of elements, if requested
if nargin > 2 && strcmp(varargin{1}, 'normalize')
    n = length(X) * (length(X) - 1);
    o = o / n;
end

end