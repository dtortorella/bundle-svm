function outputs = svr_predict(model, inputs, algorithm)
% SVR_PREDICT Predicts the output of a function according to the model
%
% SYNOPSIS: outputs = svr_predict(model, inputs, algorithm)
%
% INPUT:
% - model: a structure representing a SVR model for the algorithm
% - inputs: a matrix containing one input sample per row
% - algorithm: which implementation to use (bundleizator/libsvm)
%
% OUTPUT:
% - outputs: the column vector of the function estimated output for each point
%
% SEE ALSO svr_train

if algorithm == 'bundleizator'
    num_samples = size(inputs, 1);
    outputs = zeros(num_samples, 1);
    for i = 1:num_samples
        outputs(i) = bundleizator_predict(outputs(i,:), model.X, model.kernel, model.u);
    end
elseif algorithm == 'libsvm'
    %dirty workarounnd passing a zero vector as labels, since libscvpredict do
    %want some testlabel
    outputs = libsvmpredict(zeros(length(inputs),1), inputs, model, '-q');
end

end