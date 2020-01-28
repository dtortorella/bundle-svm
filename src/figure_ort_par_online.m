function f = figure_ort_par(name, tol, ort, par_min, par_mean, varargin)
%FIGURE_ORT_PAR creates a figure for parallelity and orthogonality
% evaluation
%SYNOPSYS:

if nargin > 5
    indices = varargin{1}
end
f = figure('Name', name, 'Position', [100 100 1000 500]);
title(name)
% yyaxis right
% if nargin > 6
%     plot(S(indices))
% else
%     plot(S)
% end
ax = get(f,'CurrentAxes');
% set(ax, 'YScale', 'log')
% ylabel 'singular values'
% yyaxis left
hold on
if nargin > 5
    plot(tol,ort(indices),'-b');
else
    plot(tol,ort,'-b');
end


if nargin > 5
    plot(tol, par_min(indices),'-m');
    plot(tol, par_mean(indices), '--m')
else
    plot(tol,par_min, '-m');
    plot(tol,par_mean, '--m')
end
ylabel 'consines'
xlabel 'tol'
set(ax,'XScale', 'log');
set(ax,'XDir', 'reverse');
legend('orthonormality', 'parallelity (min)', 'parallelity (mean)')

end

