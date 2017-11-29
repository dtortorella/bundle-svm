function output = svr_predict(model, input)
% SVR_PREDICT Predicts the output of a function according to the model
%
% SYNOPSIS: output = svr_predict(model, input)
%
% INPUT:
% - model: a structure representing a n-SVR model
% - input: a row vector of the function input point
%
% OUTPUT:
% - output: the estimated output of the function in this point
%
% SEE ALSO svr_train

%output = predict(model,input);
% dummy implementation not possible since LibSVM implementation 
% doesn't fit the current function signature: libsvmpredict expects test 
% labels in input in order to compute accurancy

end