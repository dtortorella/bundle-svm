function v = einsensitive_loss(f, y, epsilon)
%EINSENSITIVE_LOSS Computes the value of the eps-insensitive loss function
%
% SYNOPSIS: v = einsensitive_loss(f, y, epsilon)
%
% INPUT:
% - f: the value of <w,x>
% - y: the target value
% - epsilon: the threshold
%
% OUTPUT:
% - v: the eps-insensitive loss
%
% SEE ALSO einsensitive_dloss

v = max(0, abs(f - y) - epsilon);

end