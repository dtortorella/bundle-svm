function [u, t, epsilon] = bundleizator(X, y, C, kernel, loss, dloss, precision, varargin)
% BUNDLEIZATOR Implements a bundle method that solves a generic SVM
%
% SYNOPSIS: u = bundleizator(X, y, C, kernel, loss, dloss, precision)
%           [u, t] = bundleizator(X, y, C, kernel, loss, dloss, precision)
%           [...] = bundleizator(X, y, C, kernel, loss, dloss, precision, gram_svd_threshold)
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
% - gram_svd_threshold: all the singular values of the Gram matrix below
%         this threshold are discarded (optional, default 1e-6)
%
% OUTPUT:
% - u: the optimal values for the coefficients of the linear
%           combination of support vectors
% - t: the number of optimization loop iterations done
%
% SEE ALSO bundleizator_pruning, bundleizator_aggr

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

% Compute the truncated SVD of G
% this is necessary for inverse operations since G is ill-conditioned
[GU,GS,GV] = svd(G);
% discard all singular values below threshold
Gselector = diag(GS) >= gram_svd_threshold;
sGS = GS(Gselector,Gselector);
sGU = GU(:,Gselector);
sGV = GV(:,Gselector);

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
H = [];
vdloss = zeros(num_samples, 1);

%% Optimization loop
while true
    % Increment step
    t = t + 1;
    
    % Compute a_t
    % compute dloss at point u_t-1
    for i = 1:num_samples
        vdloss(i) = dloss(f(i), y(i));
    end
    
    % A(:,t) = G * vdloss / num_samples; 
    A(:,t) = (sGU * (sGS * (sGV' * vdloss))) / num_samples;
    
    % Compute b_t
    b(t,1) = Remp - A(:,t)' * u;
    
    % Update H
    % h = (A(:,t)' / G) * A;
    h = (((A(:,t)' * sGV) / sGS) * sGU') * A;
    H = [H, h(1:t-1)'; h];
    
    % Solve the dual of the quadratic master problem
    z = quadprog(0.5 * C * H, -b, -eye(t), zeros(t,1), ones(1,t), 1, [], [], [], quadprog_options);
    % Get optimal point thru dual connection
    % u_t = -0.5 * C * (G \ (A * z_t));
    u = -0.5 * C * (sGV * (sGS \ (sGU' * (A * z))));

    % Compute Remp at point u_t
    Remp = 0;
    % f = G * u;
    f = (sGU * (sGS * (sGV' * u)));
    for i = 1:num_samples
        Remp = Remp + loss(f(i), y(i));
    end
    Remp = Remp / num_samples;
    
    % Compute J(u_t)
    % J =  1/C * (u' * G * u) + Remp;
    J =  1/C * (u' * (sGU * (sGS * (sGV' * u)))) + Remp;
    
    % Evaluate J_t at point u_t
    R_t = max(u' * A + b');
    % J_t = 1/C * (u' * G * u) + R_t;
    J_t =  1/C * (u' * (sGU * (sGS * (sGV' * u)))) + R_t;
    
    % Compute epsilon
    Jmin = min(Jmin, J);
    epsilon = Jmin - J_t;
    
    % Output iteration status
    fprintf('t = %d\t Jmin = %e\t J_t(u_t) = %e\t e_t = %e\n', t, Jmin, J_t, epsilon);
    
    % Halt when we reach the desired precision
    if epsilon <= precision
        break
    end
end

end
