function [model, epsilon] = svr_train(inputs, outputs, kernel, nu, C)
% SVR_TRAIN Trains a support vector machine for regression
%
% SYNOPSIS: [model, epsilon] = svr_train(inputs, outputs, kernel, nu, C)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a column vector containing one output sample per entry
% - kernel: a function that computes the scalar product of two vectors in
% feature space. Use 'linear' | 'gaussian' | 'rbf' | 'polynomial' for
% standard implementations or give a function name of user implementation
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


epsilon = rand * nu; %by now i'm randomly choosing an epsilon value, based on the upperbound given by nu

model = fitrsvm(inputs, outputs, 'KernelFunction', kernel, 'BoxConstraint', C, 'Epsilon', epsilon);

end