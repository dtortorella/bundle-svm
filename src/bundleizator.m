function [u, sv, t, epsilon, status, U] = bundleizator(X, y, C, kernel, loss, dloss, precision, varargin)
% BUNDLEIZATOR Implements a bundle method with span vector selection that solves a generic SVM
%
% SYNOPSIS: [u, sv] = bundleizator(X, y, C, kernel, loss, dloss, precision)
%           [u, sv, t, epsilon] = bundleizator(X, y, C, kernel, loss, dloss, precision)
%           [u, sv, t, epsilon, status, U] = bundleizator(...)
%
%           [..] = bundleizator(..., 'qr')
%           [..] = bundleizator(..., 'iqr', tol)
%           [..] = bundleizator(..., 'isvd', tol)
%           [..] = bundleizator(..., 'sRRQR', f, tol)
% INPUT:
% - X: a matrix containing one sample feature vector per row
% - y: a column vector containing one sample target per entry
% - C: inverse of the regularization constant (1/lambda)
% - kernel: a function that computes the scalar product of two vectors in feature space
% - loss: a function l(f,y) that computes the loss for a single sample, taking as
%         arguments the sample target y and the scalar product f = <w,x>
% - dloss: a subgradient of the loss function with respect to f
% - precision: the required distance from optimality
% 
% - tol: the tolerance on singular values for span selection methods
% - f: the f value parameter for sRRQR factorization
%
% OUTPUT:
% - u: the optimal values for the coefficients of the linear
%           combination of support vectors
% - sv: the indices in X of the support vectors
% - t: the number of optimization loop iterations done
% - epsilon: precision reached in the last iteration
%
% - status: contains values for each algorithm step t
%     status(t,1) = epsilon;
%     status(t,2) = Jmin;
%     status(t,3) = J_t(u_t);
%     status(t,4) = J(u_t);
%     status(t,5) = R_t(u_t);
% - U: column t is the solution from the master problem at iteration t
%
% SEE ALSO bundleizator_pruning, select_span_vectors

%% Initialization
num_samples = size(X, 1);

% Master problem solver options
quadprog_options = optimoptions(@quadprog, 'Display', 'off');

% Compute Gram matrice
G = gram_matrix(X, kernel);

% Select span vectors
if nargin > 7
    switch lower(varargin{1})
        case 'qr'
            sv = select_span_vectors(G);
        case 'srrqr'
          sv = select_span_vectors(G, 'sRRQR', varargin{2}, varargin{3});
        case 'iqr'
          sv = select_span_vectors(G, 'iqr', varargin{2});
        case 'isvd'
          sv = select_span_vectors(G, 'isvd', varargin{2});
        otherwise
          error('Unknown span selection algorithm')
    end
else
     sv = select_span_vectors(G);
end

% Gamma and Gram matrix reduction given selected span vectors
Gamma = G(:,sv);
G = G(sv,sv);

num_sv = length(sv);

%% Zero-th step
t = 0;
u = zeros(num_sv,1);

% Evaluate Remp at point u_0
Remp = 0;

% f(u0) = 0
f = zeros(num_samples,1);

for i = 1:num_samples
    Remp = Remp + loss(f(i), y(i));
end
Remp = Remp / num_samples;

% the quadratic term in u_0 is 0, so J(u_0) = Remp
Jmin = Remp;

% variables initialization
A = []; % matrix of subgradients
b = []; % residual vector
H = []; % master problem matrix

% vector of values of loss derivative computation for each sample
vdloss = zeros(num_samples, 1);

%% Optimization loop
while true
    % Increment step
    t = t + 1;
    
    % Subgradient a_t computation
      
      % compute dloss at point u_t-1
    for i = 1:num_samples
        vdloss(i) = dloss(f(i), y(i));
    end
      % compute a_t
    A(:,t) = Gamma' * vdloss / num_samples;
    
    % Compute b_t
    b(t,1) = Remp - A(:,t)' * u;
    
    % Update H
    h = (A(:,t)' / G) * A;
    H = [H, h(1:t-1)'; h];
    
    % Solve the dual of the quadratic master problem
    z = quadprog(0.5 * C * H, -b, -eye(t), zeros(t,1), ones(1,t), 1, [], [], [], quadprog_options);
    % Get optimal point thru dual connection
    u = -0.5 * C * (G \ (A * z));

    % Evaluate Remp at point u_t
    f = Gamma * u;
    Remp = 0;
    for i = 1:num_samples
        Remp = Remp + loss(f(i), y(i));
    end
    Remp = Remp / num_samples;
    
    % Evaluate J at iteration point u_t
    J =  1/C * (u' * G * u) + Remp;
    
    % Compute new approximation J_t at point u_t
    R_t = max(u' * A + b');
    J_t = 1/C * (u' * G * u) + R_t;
    
    % Compute epsilon
    Jmin = min(Jmin, J);
    epsilon = Jmin - J_t;
    
    % Output iteration status
    fprintf('t = %d\t Jmin = %e\t J_t(u_t) = %e\t e_t = %e\n', t, Jmin, J_t, epsilon);
    
    status(t,1) = epsilon;
    status(t,2) = Jmin;
    status(t,3) = J_t;
    status(t,4) = J;
    status(t,5) = R_t;
    U(:,t) = u;
    
    % Halt when we reach the desired precision
    if epsilon <= precision
        break
    end
end

end
