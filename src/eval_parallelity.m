function p = eval_parallelity(X1, X2, kernel, varargin)
%EVAL_PARALLELITY Evaluates parallelity of a set of vectors X2 to a basis X1 in kernel space
% Cosine of the most orthogonal vector of X2 to the basis X1, or mean if requested.
%
% SYNOPSIS: p = eval_parallelity(X1, X2, kernel)
%           p = eval_parallelity(X1, X2, kernel, 'mean')
%
% INPUT:
% - X1: a matrix containing one sample feature vector per row of basis
% - X2: a matrix containing one sample feature vector per row of set to check
% - kernel: a function that computes the scalar product of two vectors
%           in feature space (takes row vectors)
%
% OUTPUT:
% - p: a parallelity metric, higher is better (in [0,1] range)
%
% REMARKS:
% Should be sort of an estimate of the subspace angle.

% normalized Gram matrix between X1 and X2
G = gram_norm_matrix2(X1, X2, kernel);

% pick absolute cosines
G = abs(G);

% "parallelity" of each vector X2
ps = max(G, [], 1);

% compute metric
if nargin > 3 && strcmp(varargin{1}, 'mean')
    % mean cosines of X2 vectors
    p = mean(ps);
else
    % most orthogonal vector of X2
    p = min(ps);
end

end