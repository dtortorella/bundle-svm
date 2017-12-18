function classes = svm_predict(model, features)
% SVM_PREDICT Predicts the class of a sample according to the model
%
% SYNOPSIS: classes = svm_predict(model, features)
%
% INPUT:
% - model: a structure representing a n-SVM model
% - features: a matrix containing one sample feature vector per row
%
% OUTPUT:
% - classes: a column vector of predicted class for each sample (+/-1)
%
% SEE ALSO svm_train


%dirty workarounnd passing a zero vector as labels, since libscvpredict do
%want some testlabel
classes = libsvmpredict(ones(length(features),1), features, model, '-q');

end