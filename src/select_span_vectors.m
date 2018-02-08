function sv = select_span_vectors(G)
% SELECT_SPAN_VECTORS Selects a subset of the set of samples sufficient to generate w
%
% SYNOPSIS: sv = select_span_vectors(G)
%
% INPUT:
% - G: the matrix of products of sample vectors in feature space, via the
%      kernel function (Gram matrix)
%
% OUTPUT:
% - sv: vector of indices of the selected vectors, corresponding to rows/cols of G
%
% REMARKS:
% See: Subset Selection Algorithms: Randomized vs. Deterministic, SIURO, vol 3

[~,~,p] = qr(G,0);

sv = p(1:rank(G));

end