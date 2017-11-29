function class = svm_predict(model, features)
% SVM_PREDICT Predicts the class of a sample according to the model
%
% SYNOPSIS: class = svm_predict(model, feature)
%
% INPUT:
% - model: a structure representing a n-SVM model
% - feature: a row vector of the sample features
%
% OUTPUT:
% - class: the predicted class of this sample (+/-1)
%
% SEE ALSO svm_train


%dirty workarounnd passing a zero vector as labels, since libscvpredict do
%want some testlabel
class = libsvmpredict(ones(length(features),1), features, model);

end