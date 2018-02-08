function [u, t, epsilon] = bundleizator_pruning(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
% BUNDLEIZATOR_PRUNING Implements a bundle method that solves a generic SVM, with subgradient pruning
%
% SYNOPSIS: [u] = bundleizator_pruning(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
%           [u, t, epsilon] = bundleizator_pruning(X, y, C, kernel, loss, dloss, precision, max_inactive_count, inactive_zero_threshold)
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
%           combination of sample vectors
% - t: the number of optimization loop iterations done
% - epsilon: precision reached in the last iteration
%
% REMARKS Suggested paramters for pruining are 50, 10^-7.
%
% SEE ALSO bundleizator

%% Initialization
num_samples = size(X, 1);

% Compute Gram matrices
G = gram_matrix(X, kernel);

% Master problem solver options
quadprog_options = optimoptions(@quadprog, 'Display', 'off');
H = 2/C * [0 zeros(1, num_samples); zeros(num_samples, 1) G];
h = [1 zeros(1, num_samples)];
Aineq = [];

%% Zero-th step
t = 0;
u = zeros(num_samples,1);

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
    
    A(:,t) = G * vdloss / num_samples;
    
    % Compute b_t
    b(t,1) = Remp - A(:,t)' * u;
    
    % Update Aineq
    Aineq(end+1,:) = [-1 A(:,t)'];
    
    % Solve the prima of the quadratic master problem
    [z, ~, ~, ~, mult] = quadprog(H, h, Aineq, -b, [], [], [], [], [], quadprog_options);
    u = z(2:end,1);

    % Compute Remp at point u_t
    Remp = 0;
    f = G * u;
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
    fprintf('t = %d\t Jmin = %e\t J_t(u_t) = %e\t e_t = %e\n', t, Jmin, J_t, epsilon);
    
    % Halt when we reach the desired precision
    if epsilon <= precision
        break
    end

    % Pruning
    % add new multiplier to inactive counting, update counts
    inactive_count = [inactive_count 0] + (mult.ineqlin' <= inactive_zero_threshold);
    % find which subgradients to keep
    keep = (inactive_count <= max_inactive_count);
    % discard inactive subgradients
    inactive_count = inactive_count(keep);
    A = A(:,keep);
    b = b(keep);
    Aineq = Aineq(keep,:);
end

end
