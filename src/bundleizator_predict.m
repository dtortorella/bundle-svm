function y = bundleizator_predict(x, X, kernel, u)
%BUNDLEIZATOR_PREDICT Predicts the output of a function according to 
%the linear approximation <w,x> 
%
% SYNOPSIS: y = bundleizator_classify(x, X, kernel, u)
%
% INPUT:
% - x: a row vector input point
% - X: a matrix containing one sample feature vector of the training set per row
% - kernel: a function that computes the scalar product of two vectors in feature space
% - u: a column vector of weights for the support vectors
%
% OUTPUT:
% - y: the function estimated output for the input point
%
% REMARKS w is the linear combination of support vectors in feature space with weights u
%
% SEE ALSO bundleizator, bundleizator_classify

y = 0;
for i = 1:size(X,1)
    y = y + u(i) * kernel(X(i,:), x);
end

end