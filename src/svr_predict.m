function outputs = svr_predict(model, inputs)
% SVR_PREDICT Predicts the output of a function according to the model
%
% SYNOPSIS: outputs = svr_predict(model, inputs)
%
% INPUT:
% - model: a structure representing a n-SVR model
% - inputs: a matrix containing one input sample per row
%
% OUTPUT:
% - outputs: the column vector of the function estimated output for each point
%
% SEE ALSO svr_train


%dirty workarounnd passing a zero vector as labels, since libscvpredict do
%want some testlabel
outputs = libsvmpredict(zeros(length(inputs),1), inputs, model);

end