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

model = fitcsvm(features,classes,'KernelFunction', kernel);

end