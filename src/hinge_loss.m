function v = hinge_loss(f, y)
%HINGE_LOSS Computes the value of the hinge loss function
%
% SYNOPSIS: v = hinge_loss(f, y)
%
% INPUT:
% - f: the value of <w,x>
% - y: the target value
%
% OUTPUT:
% - v: the hinge loss
%
% SEE ALSO hinge_dloss

v = max(0, 1 - y * f);

end