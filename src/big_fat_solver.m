function [u, sv, J] = big_fat_solver(X, y, C, kernel, precision, loss, varargin)
%BIG_FAT_SOLVER Quadratic program solver for a SVM/SVR
%
% SYNOPSIS: [u, sv] = big_fat_solver(X, y, C, kernel, precison, 'hinge')
%           [u, sv] = big_fat_solver(X, y, C, kernel, precision, 'einsensitive', eps)
%
% INPUT:
% - X: a matrix containing one sample feature vector per row
% - y: a column vector containing one sample target per entry
% - C: inverse of the regularization constant (1/lambda)
% - precision: how close to the optimal value of J we should get
% - kernel: a function that computes the scalar product of two vectors in feature space
% - loss: type of loss, can be either 'hinge' for SVM or 'einsensitive' for SVR
%
% OUTPUT:
% - u: the optimal values for the coefficients of the linear
%           combination of support vectors
% - sv: the indices in X of the support vectors
% - J: the function minimum value
%
% SEE ALSO quadprog, bundleizator

%% Initialization
m = size(X, 1);

% QP solver options
quadprog_options = optimoptions(@quadprog, 'Display', 'iter', 'OptimalityTolerance', precision);

% Get the SVs, and compute Gram matrices
G = gram_matrix(X, kernel);
sv = select_span_vectors(G);

GX = G(:,sv);
G = G(sv,sv);

n = length(sv);

%% Setup constraints
if strcmpi(loss, 'hinge')
    % Hinge loss for SVM
    A = [zeros(m,n), -eye(m); bsxfun(@times, GX, y), -eye(m)];
    b = [zeros(m,1); -ones(m,1)];
else
    % Epsilon-insensitive loss for SVR
    eps = varargin{1};
    A = [zeros(m,n), -eye(m); GX, -eye(m); -GX, -eye(m)];
    b = [zeros(m,1); y + eps; -y + eps];
end

%% Solve QP
H = [G * 2 / C, zeros(n,m); zeros(m,n+m)];
g = [zeros(n,1); ones(m,1) / m];

[x, J] = quadprog(H, g, A, b, [], [], [], [], [], quadprog_options);
u = x(1:n);

end
