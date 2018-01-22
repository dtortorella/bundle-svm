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
% - precision: the required distance from optimality (optional, only for bundleizator)
%
% OUTPUT:
% - model: a structure representing the SVR model for this algorithm
%
% SEE ALSO svm_train, svr_predict

%LibSVM has the following kernels that can be selected using an integer
%
%	0 -- linear: u'*v
%	1 -- polynomial: (gamma*u'*v + coef0)^degree
%	2 -- radial basis function: exp(-gamma*|u-v|^2)
%	3 -- sigmoid: tanh(gamma*u'*v + coef0)
%	4 -- precomputed kernel (kernel values in training_instance_matrix)
%
%   (default is 2)

% since the options are a string here's the code that create the appropriate one using the given function parameters:
%   -s 4   sets the nu-svr model
%   -n nu  sets the nu hyperparameter to nu
%   -c C   sets the C hyperparameter to C 
%   -t k   sets the k-th kernel function type as descripted above

if algorithm == 'bundleizator'
    model.X = inputs;
    model.kernel = kernel;
    if isempty(varargin)
        model.u = bundleizator(inputs, outputs, C, kernel, ...
            @(f,y) einsensitive_loss(f, y, epsilon), @(f,y) einsensitive_dloss(f, y, epsilon));
    else
        model.u = bundleizator(inputs, outputs, C, kernel, ...
            @(f,y) einsensitive_loss(f, y, epsilon), @(f,y) einsensitive_dloss(f, y, epsilon), varargin{1});
    end
elseif algorithm == 'libsvm'
    options = sprintf('-s 3 -c %f -p %f %s -q ', C, epsilon, kernel);
    model = libsvmtrain(outputs, inputs, options);
end

end