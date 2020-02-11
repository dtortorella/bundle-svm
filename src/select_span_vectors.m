function sv = select_span_vectors(G, varargin)
% SELECT_SPAN_VECTORS Selects a subset of the set of samples sufficient to generate w
%
% SYNOPSIS: sv = select_span_vectors(G)
%           sv = select_span_vectors(G, algorithm, params...)
%
% INPUT:
% - G: the matrix of products of sample vectors in feature space, via the
%      kernel function (Gram matrix)
% - algorithm: the algorithm to use to find the spanning set
%            - 'qr'(default) Matlab's qr+rank. 
%                 No params required.
%            - 'srrqr' strong Rank Revealing QR. 
%                 f is the strong qr factor, tol is the 
%            - 'iqr' incremental selection with qr factorization.
%                 tol is the threshold to discard R eigenvalues
%            - 'isvd' incremental selection with svd factorization.
%                 tol is the threshold to discard singular values
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
        [~,~,sv] = sRRQR_tol(G, varargin{2}, varargin{3});
    case 'iqr'
        % repeat qr adding one sample at a time, estimating rank
        [~,~,~,sv] = iqr(G, varargin{2});
    case 'isvd'
        % repeat svd arring one sample at a time, estmating rank
        [~,~,sv] = isvd(G, varargin{2});
otherwise
    error('Unknown algorithm')
end

end