function v = einsensitive_dloss(f, y, epsilon)
%EINSENSITIVE_DLOSS Computes the value of the derivative 
%of the eps-insensitive loss function w.r.t. f
%
% SYNOPSIS: v = einsensitive_dloss(f, y, epsilon)
%
% INPUT:
% - f: the value of <w,x>
% - y: the target value
% - epsilon: the threshold
%
% OUTPUT:
% - v: the derivative w.r.t. f of the eps-insensitive loss function 
%
% SEE ALSO einsensitive_loss

if abs(f - y) <= epsilon
    v = 0;
else
    v = sign(f - y);
end

end