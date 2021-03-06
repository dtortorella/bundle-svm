function [u_star, iterations] = bundleizator_aggr(bundle_size, X, y, C, kernel, loss, dloss, precision, varargin)
%BUNDLEIZATOR Implements a bundle method that solves a generic SVM
%
% SYNOPSIS: u_star = bundleizator(X, y, C, kernel, loss, dloss, precision)
%           [u_star, iterations] = bundleizator(X, y, C, kernel, loss, dloss, precision)
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
%                       this threshold are discarded (optional, default 1e-6)
%
% OUTPUT:
% - u_star: the optimal values for the coefficients of the linear
%           combination of support vectors
% - iterations: the number of optimization loop iterations done
%
% SEE ALSO bundleizator, bundleizator_pruning

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
invG = sGV * inv(sGS) * sGU';

%% Zero-th step
t = 0;
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
    A(:,t+1) = G * vdloss / num_samples;
    
    % Compute b_t+1
    b(t+1,1) = Remp - A(:,t+1)' * u_t;
    
    % Evaluate J_t+1 at point u_t
    R_t1 = max(u_t' * A + b');
    J_t1 = 1/C * (u_t' * G * u_t) + R_t1;
    
    % Compute epsilon
    Jmin = min(Jmin, J_t1);
    epsilon = Jmin - J_t;
    
    % Output iteration status
    fprintf('t = %d\t Remp = %e\t Jt = %e\t J_t+1 = %e\t e_t = %e\n', t, Remp, J_t, J_t1, epsilon);
    
    % Halt when we reach the desired precision
    if epsilon <= precision
        break
    end
    
    % Compute aggregated sub-gradients after t > bundle_size
    if t < bundle_size
        aggrA = A;
        aggrb = b;
        
        % Update H
        % h = (A(:,t+1)' / G) * A;
        h = (((A(:,t+1)' * sGV) / sGS) * sGU') * A;
        H = [H, h(1:t)'; h];
    else
        fprintf('Aggregation\n');
        %aggrA(:,1) = (z_t(1) * aggrA(:,1) + z_t(2) * aggrA(:,2)) / (z_t(1) + z_t(2));
        a_hat = zeros(size(aggrA, 1), 1);
        for i = 1:length(z_t)
            a_hat = a_hat + z_t(i) * aggrA(:,i);
        end
        aggrA(:,1) = a_hat;
        aggrA(:,2:end) = A(:,end-(bundle_size-2):end);
        %aggrb(1,1) = (z_t(1) * aggrb(1,1) + z_t(2) * aggrb(2,1)) / (z_t(1) + z_t(2));
        b_hat = 0;
        for i = 1:length(z_t)
            b_hat = b_hat + b(i);
        end
        aggrb(1,1) = b_hat;
        aggrb(2:end,1) = b(end-(bundle_size-2):end,1);
        
        % Compute H from aggregation
        h1 = aggrA' * (sGV * (sGS \ (sGU' * aggrA(:,1))));
        h2 = aggrA' * (sGV * (sGS \ (sGU' * aggrA(:,end))));
        H(2:end-1,2:end-1) = H(3:end,3:end);
        H(:,1) = h1;
        H(1,2:end) = h1(2:end)';
        H(2:end,end) = h2(2:end);
        H(end,2:end-1) = h2(2:end-1)';
    end
    
    % Increment step
    t = t + 1;
    
    % Solve the dual of the quadratic subproblem
%     dim = min(bundle_size, t);
    dim = length(aggrb);
    z_t = quadprog(0.5 * C * H, -aggrb, -eye(dim), zeros(dim,1), ones(1,dim), 1, [], [], [], quadprog_options);
    % Get optimal point thru dual connection
    % u_t = -0.5 * C * (G \ (A * z_t));
    u_t = -0.5 * C * (sGV * (sGS \ (sGU' * (aggrA * z_t))));
    
    % Evaluate J_t at point u_t
    R_t = max(u_t' * aggrA + aggrb');
    J_t = 1/C * (u_t' * G * u_t) + R_t;
end

%% Function outputs

% Optimal value of u
u_star = u_t;

% Number of iterations, if requested
if nargout == 2
    iterations = t;
end

end
