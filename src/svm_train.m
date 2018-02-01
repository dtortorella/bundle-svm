function model = svm_train(features, classes, kernel, C, algorithm, varargin)
% SVM_TRAIN Trains a support vector machine for classification
%
% SYNOPSIS: model = svm_train(features, classes, kernel, C, algorithm)
%           model = svm_train(features, classes, kernel, C, 'bundleizator', precision)
%
% INPUT:
% - features: a matrix containing one sample feature vector per row
% - classes: a column vector containing one sample class per entry, must be +/-1
% - kernel: a function that computes the scalar product of two vectors in feature space
% - C: hyperparameter, a non-negative regularization constant
% - algorithm: which implementation to use (bundleizator/libsvm)
% - precision: the required distance from optimality (optional, default 10^-6, only for bundleizator)
%
% OUTPUT:
% - model: a structure representing the SVM model for this algorithm
%
% SEE ALSO svr_train, svm_predict

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
    model.X = features;
    model.kernel = kernel;
    if isempty(varargin)
        model.u = bundleizator_pruning(features, classes, C, kernel, @hinge_loss, @hinge_dloss, 1e-6, 50, 1e-7);
    else
        model.u = bundleizator_pruning(features, classes, C, kernel, @hinge_loss, @hinge_dloss, varargin{1}, 50, 1e-7);
    end
elseif strcmp(algorithm, 'libsvm')
    options = sprintf('-s 0 -c %f %s -q', C, kernel);
    model = libsvmtrain(classes, features, options);
end

end