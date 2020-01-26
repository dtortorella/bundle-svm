function sv = select_span_vectors(G, varargin)
% SELECT_SPAN_VECTORS Selects a subset of the set of samples sufficient to generate w
%
% SYNOPSIS: sv = select_span_vectors(G)
%           sv = select_span_vectors(G, algorithm, ...)
%
% INPUT:
% - G: the matrix of products of sample vectors in feature space, via the
%      kernel function (Gram matrix)
% - algorithm: the algorithm to use to find the spanning set (default MATLAB qr+rank)
%
% OUTPUT:
% - sv: vector of indices of the selected vectors, corresponding to rows/cols of G
%
% REMARKS:
% See: Subset Selection Algorithms: Randomized vs. Deterministic, SIURO, vol 3

if nargin > 1
    algorithm = varargin{1};
else
    algorithm = 'qr';
end

switch lower(algorithm)
    case 'qr'
        % QR with pivoting, with rank estimation provided by MATLAB
        [~,~,p] = qr(G,0);
        sv = p(:,1:rank(G));
    case 'srrqr'
        % sRRQR by Gu & Eisenstat
        [~,~,sv] = sRRQR(G, varargin{2}, varargin{3}, varargin{4});
    case 'online_qr'
        % repeat qr adding one sample at a time, estimating rank
        [~,~,~,sv] = online_qr(G, varargin{2});
    case 'online_svd'
        % repeat svd arring one sample at a time, estmating rank
        [~,~,sv] = online_svd(G, varargin{2});
otherwise
    error('Unknown algorithm')
end

end