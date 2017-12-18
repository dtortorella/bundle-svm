function model = svm_train(features, classes, kernel, nu)
% SVM_TRAIN Trains a support vector machine for classification
%
% SYNOPSIS: model = svm_train(features, classes, kernel, nu)
%
% INPUT:
% - features: a matrix containing one sample feature vector per row
% - classes: a column vector containing one sample class per entry, must be +/-1
% - kernel: a function that computes the scalar product of two vectors in feature space
% - nu: hyperparameter, fraction of support vectors (between 0 and 1)
%
% OUTPUT:
% - model: a structure representing a n-SVM model
%
% REMARKS The optimization algorithm is based on bundle methods
%
% SEE ALSO svr_train, svm_predict

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
%   -s 1   sets the nu-svm model
%   -n nu  sets the nu value of the model to nu
%   -t k   sets the k-th kernel function type as descripted above

options = sprintf('-s 1 -n %f -t %d -q', nu, kernel);

model = libsvmtrain(classes, features, options);

end