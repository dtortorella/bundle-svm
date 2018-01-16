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
% - dloss: the derivative of the loss function with respect to f
% - precision: the required distance from optimality
% 
% OUTPUT:
% - u_star: the optimal values for the coefficients of the linear
%           combination of support vectors

%% Initialization
num_samples = size(X, 2);

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
  
Ginv = inv(G);

% Zero-th step
% assuming a_t = 0, b_t = 0, u_t = 0
Jt_prev = 0;

Jt = 

z = quadprog();

F = z;
u = F z;

Jt = J(u);
Jmin = min(Jmin, Jt+1) - Jt;





end

