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


%dirty workarounnd passing a zero vector as labels, since libscvpredict do
%want some testlabel
output = libsvmpredict(zeros(length(input),1), input, model);

end