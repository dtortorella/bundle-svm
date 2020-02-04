function [u, sv, t, epsilon, status] = bundleizator_pruning(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
% BUNDLEIZATOR_PRUNING Implements a bundle method that solves a generic SVM, with subgradient pruning
%
% SYNOPSIS: [u, sv] = bundleizator_pruning(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
%           [u, sv, t, epsilon, status] = bundleizator_pruning(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
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
%
% OUTPUT:
% - u: the optimal values for the coefficients of the linear
%           combination of support vectors
% - sv: the indices in X of the support vectors
% - t: the number of optimization loop iterations done
% - epsilon: precision reached in the last iteration
% - status: status(:,1) contains the epsilon value at each iteration
%           status(:,2) contains the bundle set dimension at each iteration
%
% REMARKS Suggested paramters for pruining are 50, 10^-7.
%
% SEE ALSO bundleizator

%% Initialization
num_samples = size(X, 1);

% Master problem solver options
quadprog_options = optimoptions(@quadprog, 'Display', 'off');

% Get the SVs, and compute Gram matrices
G = gram_matrix(X, kernel);
sv = select_span_vectors(G);

GX = G(:,sv);
G = G(sv,sv);

num_sv = length(sv);

%% Zero-th step
t = 0;
u = zeros(num_sv,1);

% Compute Remp at point u_0
Remp = 0;
f = zeros(num_samples,1);
for i = 1:num_samples
    Remp = Remp + loss(f(i), y(i));
end
Remp = Remp / num_samples;

% the quadratic term in u_0 is 0, so J(u_0) = Remp
Jmin = Remp;

% variables initialization
A = [];
b = [];
H = [];
vdloss = zeros(num_samples, 1);
inactive_count = [];

%% Optimization loop
while true
    % Increment step
    t = t + 1;
    
    % Compute a_t
    % compute dloss at point u_t-1
    for i = 1:num_samples
        vdloss(i) = dloss(f(i), y(i));
    end
    
    A(:,end+1) = GX' * vdloss / num_samples;
    
    % Compute b_t
    b(end+1,1) = Remp - A(:,end)' * u;
    
    % Update H
    h = (A(:,end)' / G) * A;
    H = [H, h(1:end-1)'; h];
    
    % Solve the dual of the quadratic master problem
    dim = length(b);
    z = quadprog(0.5 * C * H, -b, -eye(dim), zeros(dim,1), ones(1,dim), 1, [], [], [], quadprog_options);
    % Get optimal point thru dual connection
    u = -0.5 * C * (G \ (A * z));

    % Compute Remp at point u_t
    Remp = 0;
    f = GX * u;
    for i = 1:num_samples
        Remp = Remp + loss(f(i), y(i));
    end
    Remp = Remp / num_samples;
    
    % Compute J(u_t)
    J =  1/C * (u' * G * u) + Remp;
    
    % Evaluate J_t at point u_t
    R_t = max(u' * A + b');
    J_t = 1/C * (u' * G * u) + R_t;
    
    % Compute epsilon
    Jmin = min(Jmin, J);
    epsilon = Jmin - J_t;
    
    % Output iteration status
    fprintf('t = %d (%d subgradients)\t Jmin = %e\t J_t(u_t) = %e\t e_t = %e\n', t, dim, Jmin, J_t, epsilon);
    status(t,1) = epsilon;
    status(t,2) = dim;
    
%     model.u = u;
%     model.kernel = kernel;
%     pred = svm_predict(model, X, 'bundleizator');
%     acc = mean( pred == y);
%     status(t,2) = acc;

    % Halt when we reach the desired precision
    if epsilon <= precision
        break
    end

    % Pruning
    % add new multiplier to inactive counting, update counts
    inactive_count = [inactive_count 0] + (z' <= inactive_zero_threshold);
    % find which subgradients to keep
    keep = (inactive_count <= max_inactive_count);
    % discard inactive subgradients
    inactive_count = inactive_count(keep);
    A = A(:,keep);
    b = b(keep);
    H = H(keep,keep);
end

end
