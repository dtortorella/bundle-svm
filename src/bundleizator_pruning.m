function [u_star, iterations] = bundleizator_pruning(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold, varargin)
% BUNDLEIZATOR_PRUNING Implements a bundle method that solves a generic SVM, with subgradient pruning
%
% SYNOPSIS: u_star = bundleizator(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
%           [u_star, iterations] = bundleizator(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
%           [...] = bundleizator(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold, gram_svd_threshold)
%
% INPUT:
% - X: a matrix containing one sample feature vector per row
% - y: a column vector containing one sample target per entry
% - C: inverse of the regularization constant (1/lambda)
% - kernel: a function that computes the scalar product of two vectors in feature space
% - loss: a function l(f,y) that computes the loss for a single sample, taking as
%         arguments the sample target y and the scalar product f = <w,x>
% - dloss: a subgradient of the loss function with respect to f
% - precision: the required distance from optimality
% - max_inactive_count: for how many iterations a subgradient can be
%         inactive in the master problem before being discarded
% - inactive_zero_threshold: subgradients with multiplier below this
%         threshold will be considered inactive in the current iteration
% - gram_svd_threshold: all the singular values of the Gram matrix below
%         his threshold are discarded (optional, default 1e-6)
%
% OUTPUT:
% - u_star: the optimal values for the coefficients of the linear
%           combination of support vectors
% - iterations: the number of optimization loop iterations done
%
% REMARKS Suggested paramters for pruining are 50, 10^-7.
%
% SEE ALSO bundleizator, bundleizator_aggr

%% Initialization
num_samples = size(X, 1);

if ~isempty(varargin)
    gram_svd_threshold = varargin{1};
else
    gram_svd_threshold = 1e-6;
end

% Master problem solver options
quadprog_options = optimoptions(@quadprog, 'Display', 'off');

% Compute the Gram matrix
G = gram_matrix(X, kernel);

% Compute the reduced SVD of G
% this is necessary for inverse operations since G is ill-conditioned
[GU,GS,GV] = svd(G);
% discard all singular values below threshold
Gselector = diag(GS) >= gram_svd_threshold;
sGS = GS(Gselector,Gselector);
sGU = GU(:,Gselector);
sGV = GV(:,Gselector);

%% Zero-th step
t = 0;
dim = 0;
% we take a_0, b_0 = 0

% since the regularizer function is quadratic, its minimum is 0
u_t = zeros(num_samples,1);
J_t = 0;

% no J_0(u_-1), so to make min(Jmin, J_1(u_0)) work...
Jmin = Inf;

A = [];
b = [];
H = [];
vdloss = zeros(num_samples, 1);
inactive_count = [];

%% Optimization loop
while true
    % Compute Remp and dloss at point u_t
    Remp = 0;
    f = G * u_t;
    for i = 1:num_samples
        vdloss(i) = dloss(f(i), y(i));
        Remp = Remp + loss(f(i), y(i));
    end
    Remp = Remp / num_samples;
    
    % Compute a_t+1
    A(:,end+1) = G * vdloss / num_samples;
    
    % Compute b_t+1
    b(end+1,1) = Remp - A(:,end)' * u_t;
    
    % Evaluate J_t+1 at point u_t
    R_t1 = max(u_t' * A + b');
    J_t1 = 1/C * (u_t' * G * u_t) + R_t1;
    
    % Compute epsilon
    Jmin = min(Jmin, J_t1);
    epsilon = Jmin - J_t;
    
    % Output iteration status
    fprintf('t = %d (%d subgradients)\t Remp = %e\t J_t = %e\t J(u_t) = %e\t e_t = %e\n', t, dim, Remp, J_t, J_t1, epsilon);
    
    % Halt when we reach the desired precision
    if epsilon <= precision
        break
    end
    
    % Update H
    % h = (A(:,t+1)' / G) * A;
    h = (((A(:,end)' * sGV) / sGS) * sGU') * A;
    H = [H, h(1:end-1)'; h];
    
    % Increment step
    t = t + 1;
    
    % Solve the dual of the quadratic master problem
    dim = length(b);
    z_t = quadprog(0.5 * C * H, -b, -eye(dim), zeros(dim,1), ones(1,dim), 1, [], [], [], quadprog_options);
    % Get optimal point thru dual connection
    % u_t = -0.5 * C * (G \ (A * z_t));
    u_t = -0.5 * C * (sGV * (sGS \ (sGU' * (A * z_t))));
    
    % Evaluate J_t at point u_t
    R_t = max(u_t' * A + b');
    J_t = 1/C * (u_t' * G * u_t) + R_t;
    
    % Pruning
    % add new multiplier to inactive counting, update counts
    inactive_count = [inactive_count 0] + (z_t' <= inactive_zero_threshold);
    % find which subgradients to keep
    keep = (inactive_count <= max_inactive_count);
    % discard inactive subgradients
    inactive_count = inactive_count(keep);
    A = A(:,keep);
    b = b(keep);
    H = H(keep,keep);
end

%% Function outputs

% Optimal value of u
u_star = u_t;

% Number of iterations, if requested
if nargout == 2
    iterations = t;
end

end
