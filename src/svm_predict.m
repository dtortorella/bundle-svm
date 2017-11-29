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

%class = predict(model,features);
% dummy implementation not possible since LibSVM implementation 
% doesn't fit the current function signature: libsvmpredict expects test 
% labels in input in order to compute accurancy

end