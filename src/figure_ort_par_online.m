function f = figure_ort_par_online(name, tol, nsv, ort, par_min, par_mean, varargin)
%FIGURE_ORT_PAR_online creates a figure for parallelity and orthogonality
% evaluation of online span selection methods
%SYNOPSYS:

indices_bool = 0;
if nargin > 6
    indices = varargin{1};
    indices_bool = 1;
end

f = figure('Name', name, 'Position', [100 100 600 400]);
title(name)

yyaxis right
if indices_bool
    plot(nsv(indices))
else
    plot(tol,nsv)
end
ax = get(f,'CurrentAxes');

ylabel 'selected span vectors num.'
yyaxis left
hold on
if indices_bool
    plot(tol,ort(indices),'-b');
else
    plot(tol,ort,'-b');
end


if indices_bool
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

