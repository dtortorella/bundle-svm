function v = hinge_dloss(f,y)
%HINGE_DLOSS Computes the value of the derivative 
%of the hinge loss function w.r.t. f
%
% SYNOPSIS: v = hinge_dloss(f, y)
%
% INPUT:
% - f: the value of <w,x>
% - y: the target value
%
% OUTPUT:
% - v: the derivative w.r.t. f of the hinge loss function 
%
% SEE ALSO hinge_loss

if y * f >= 1
    v = 0;
else
    v = -y;
end

end