function classes = svm_predict(model, features, algorithm)
% SVM_PREDICT Predicts the class of a sample according to the model
%
% SYNOPSIS: classes = svm_predict(model, features, algorithm)
%
% INPUT:
% - model: a structure representing a SVM model for the algorithm
% - features: a matrix containing one sample feature vector per row
% - algorithm: which implementation to use (bundleizator/libsvm)
%
% OUTPUT:
% - classes: a column vector of predicted class for each sample (+/-1)
%
% SEE ALSO svm_train

if algorithm == 'bundleizator'
    num_samples = size(features, 1);
    classes = zeros(num_samples, 1);
    for i = 1:num_samples
        classes(i) = bundleizator_classify(features(i,:), model.X, model.kernel, model.u);
    end
elseif algorithm == 'libsvm'
    %dirty workarounnd passing a zero vector as labels, since libscvpredict do
    %want some testlabel
    classes = libsvmpredict(ones(length(features),1), features, model, '-q');
end

end