function G = givens(x1,x2)
%GIVENS appy rotation to the input vectors
%   wrapper for missing givens function in sRRQR repository.
[G,~] = planerot([x1;x2]);
end

