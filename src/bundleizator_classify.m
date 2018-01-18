function y = bundleizator_classify(x, X, kernel, u)
%BUNDLEIZATOR_CLASSIFY Computes the predicted class for the input sample 
%given the support vector weights
%
% SYNOPSIS: y = bundleizator_classify(x, X, kernel, u)
%
% INPUT:
% - x: a row vector sample
% - X: a matrix containing one sample feature vector of the training set per row
% - kernel: a function that computes the scalar product of two vectors in feature space
% - u: a column vector of weights for the support vectors
%
% OUTPUT:
% - y: the predicted class for the input sample (+/-1)
%
% SEE ALSO bundleizator, bundleizator_predict

s = 0;
for i = 1:size(X,1)
    s = s + u(i) * kernel(X(i,:), x);
end
y = sign(s);

end