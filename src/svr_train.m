function model = svr_train(inputs, outputs, kernel, C, epsilon, algorithm, varargin)
% SVR_TRAIN Trains a support vector machine for regression
%
% SYNOPSIS: model = svr_train(inputs, outputs, kernel, C, epsilon, algorithm)
%           model = svr_train(inputs, outputs, kernel, C, epsilon, 'bundleizator', precision)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a column vector containing one output sample per entry
% - kernel: a function that computes the scalar product of two vectors in feature space
% - C: hyperparameter, a non-negative regularization constant
% - epsilon: hyperparameter, for the epsilon-insensitive loss
% - algorithm: which implementation to use (bundleizator/libsvm)
% - precision: the required distance from optimality (optional, default 10^-6, only for bundleizator)
%
% OUTPUT:
% - model: a structure representing the SVR model for this algorithm
%
% SEE ALSO svm_train, svr_predict

% LibSVM has the following kernels that can be selected using an integer
%  -t k   sets the k-th kernel function type as descripted below:
%	0 -- linear: u'*v
%	1 -- polynomial: (gamma*u'*v + coef0)^degree
%	2 -- radial basis function: exp(-gamma*|u-v|^2)
%	3 -- sigmoid: tanh(gamma*u'*v + coef0)
%	4 -- precomputed kernel (kernel values in training_instance_matrix)
%
%   (default is 2)
% see libsvmtrain for more information

if strcmp(algorithm, 'bundleizator')
    model.X = inputs;
    model.kernel = kernel;
    if isempty(varargin)
        model.u = bundleizator_pruning(inputs, outputs, C, kernel, ...
            @(f,y) einsensitive_loss(f, y, epsilon), @(f,y) einsensitive_dloss(f, y, epsilon), 1e-6, ...
            50, 1e-7);
    else
        model.u = bundleizator_pruning(inputs, outputs, C, kernel, ...
            @(f,y) einsensitive_loss(f, y, epsilon), @(f,y) einsensitive_dloss(f, y, epsilon), varargin{1}, ...
            50, 1e-7);
    end
elseif strcmp(algorithm, 'libsvm')
    options = sprintf('-s 3 -c %f -p %f %s -q ', C, epsilon, kernel);
    model = libsvmtrain(outputs, inputs, options);
end

end