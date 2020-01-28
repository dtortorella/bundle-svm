function G = givens(x1,x2)
%GIVENS 
%   wrapper for missing givens function in sRRQR
[G,~] = planerot([x1;x2]);
end

