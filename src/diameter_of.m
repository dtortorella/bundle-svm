function diameter = diameter_of(X)
% DIAMETER_OF Computes the diamter of a set of points
%
% SYNOPSIS: diameter = diameter_of(X)
%
% INPUT:
% - X: a matrix containing one point per row
%
% OUTPUT:
% - diameter: the maximum distance between pair of points

[elements, dimentions] = size(X);
distances = zeros(elements, elements);

for k = 1:dimentions
    distances = distances + (X(:,k) - X(:,k)').^2;
end

diameter = sqrt(max(max(distances)));

end