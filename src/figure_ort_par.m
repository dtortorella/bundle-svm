function f = figure_ort_par(name, S, ort, par_min, par_mean, varargin)
%FIGURE_ORT_PAR creates a figure for parallelity and orthogonality
% evaluation
%SYNOPSYS:

if nargin == 6
    indices = varargin{1}
end
f = figure('Name', name, 'Position', [100 100 600 400]);
title(name,'Interpreter','Latex')
yyaxis right
if nargin == 6
    plot(S(indices))
else
    plot(S)
end
ax = get(f,'CurrentAxes');
set(ax, 'YScale', 'log')
ylabel 'singular values'
yyaxis left
hold on
if nargin == 6
    plot(ort(indices),'-b');
else
    plot(ort,'-b');
end


if nargin == 6
    plot(par_min(indices),'-m');
    plot(par_mean(indices), '--m')
else
    plot(par_min, '-m');
    plot(par_mean, '--m')
end
ylabel 'consines'
xlabel 'rank'
legend('orthonormality', 'parallelity (min)', 'parallelity (mean)', 'singular values');

end

