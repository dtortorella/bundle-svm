function [model, epsilon] = svr_train(inputs, outputs, kernel, nu, C)
% SVR_TRAIN Trains a support vector machine for regression
%
% SYNOPSIS: [model, epsilon] = svr_train(inputs, outputs, kernel, nu, C)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a column vector containing one output sample per entry
% - kernel: a function that computes the scalar product of two vectors in feature space
% - nu: hyperparameter, fraction of support vectors (between 0 and 1)
% - C: hyperparameter, a non-negative regularization constant
%
% OUTPUT:
% - model: a structure representing a n-SVR model
% - epsilon: the accuracy of the model (as in an e-SVR)
%
% REMARKS The optimization algorithm is based on bundle methods
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

options = sprintf('-s 4 -n %f -c %f -t %d -q ', nu, C, kernel);

model = libsvmtrain(outputs, inputs, options);

%by now i'm randomly choosing an epsilon value, based on the upperbound
%given by nu becouse i can't figure it out how to retrieve it from the
%model. libsvmpredict returns an 'epsilon' value btw...
epsilon = rand * nu; %(it's weird , i know, but 'rand' is actually a function call)

end