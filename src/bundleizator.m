function u_star = bundleizator(X, y, C, kernel, loss, dloss, precision)
%BUNDLEIZATOR Implements a bundle method that solves a generic SVM
%
% SYNOPSIS: u_star = bundleizator(X, y, C, kernel, loss, dloss, precision)
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
%
% OUTPUT:
% - u_star: the optimal values for the coefficients of the linear
%           combination of support vectors

%% Initialization
num_samples = size(X, 1);

% Compunte the Gram matrix
G = zeros(num_samples);
for i = 1:num_samples
    j = 1;
    while j < i
        G(i,j) = kernel(X(i,:), X(j,:));
        G(j,i) = G(i,j);
        j = j + 1;
    end
    G(i,i) = kernel(X(i,:), X(i,:));
end

% Compute the reduced SVD of G
% this is necessary for inverse operations since G is ill-conditioned
[GU,GS,GV] = svd(G);
% discard all singular values below precision
Gselector = diag(GS) > 1e-6;
sGS = GS(Gselector,Gselector);
sGU = GU(:,Gselector);
sGV = GV(:,Gselector);

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

%% Optimization loop
while true
    % Update a_t
    vdloss = zeros(num_samples, 1);
    f = G * u_t;
    for i = 1:num_samples
        vdloss(i) = dloss(f(i), y(i));
    end
    A(:,t+1) = G * vdloss;
    
    % Update b_t
    Remp = 0;
    for i = 1:num_samples
        Remp = Remp + loss(f(i), y(i));
    end
    Remp = Remp / num_samples;
    b(t+1) = Remp - A(:,t+1)' * u_t;
    
    % Evaluate J_t+1 at point u_t
    R_t1 = max(u_t' * A + b(t+1));
    J_t1 = 1/C * (u_t' * G * u_t) + R_t1;
    %J_t1 = (u_t' * G * u_t) + C * Remp;
    
    % Compute epsilon
    Jmin = min(Jmin, J_t1);
    epsilon = Jmin - J_t;

    fprintf('t = %d \t Remp = %f \t e_t %f \n', t, Remp, epsilon);
    
    % Halt when we reach the desired precision
    if epsilon <= precision
        break
    end
    
    % Update H
    % h = (A(:,t+1)' / G) * A;
    h = (((A(:,t+1)' * sGV) / sGS) * sGU') * A;
    H = [H, h(1:t)'; h];
    
    % Increment step
    t = t + 1;
    
    % Solve the dual of the quadratic subproblem
    z_t = quadprog(0.5 * C * H, -b, -eye(t), zeros(t,1), ones(1,t), 1);
    % Get optimal point thru dual connection
    % u_t = -0.5 * C * (G \ (A * z_t));
    u_t = -0.5 * C * (sGV * (sGS \ (sGU' * (A * z_t))));
    
    % Evaluate J at point u
    R_t = max(u_t' * A + b');
    J_t = 1/C * (u_t' * G * u_t) + R_t;
end

% Optimal value of u
u_star = u_t;

end
