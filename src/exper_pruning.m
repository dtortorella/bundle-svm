%% load dataset
mlcup_training_set = importdata("../data/ml-cup17.train.csv");
X = mlcup_training_set(:,2:end-2);
y1 = mlcup_training_set(:,end-1);

%%
kernel = @(x,y) exp(-.1*norm(x-y)^2);

t= [];
eps = [];
stat = [];
%
epsilons = logspace(-5,5,20);
Cs = logspace(2,6,5);

f = figure('Name','pruning iteration vs precision. varing C', ...
    'Position', [100 100 1000 600]);
%% params
C = 1e5;
epsilon = 0.2;
precision = 1e-6;


%%
us = {};
usd = {};
stats = {};
times = [];

inac_zero_thres = 1e-4;

k = [5, 10, 25, 50, 100, 250, 500];

%% non pruning values
tic;
[u_non_pruning,~,~,~,stat_non_pruning] = bundleizator(X, y1, C, kernel, ...
      @(f,Y) einsensitive_loss(f, Y, epsilon), @(f,Y) einsensitive_dloss(f, Y, epsilon),...
      precision);
time_non_pruning = toc; 
%%

for i = 1:7
    tic;
    [u,~,~,~,stat] = bundleizator_pruning(X, y1, C, kernel, ...
      @(f,Y) einsensitive_loss(f, Y, epsilon), @(f,Y) einsensitive_dloss(f, Y, epsilon),...
      precision, k(i), inac_zero_thres);
  times(i) = toc;
  us{i} = u;
  usd{i} = norm(u-u_non_pruning);
  stats{i} = stat;
end

%%



%% plot z
plot(sort(status, 'descend'))
set(gca, 'YScale', 'log');

%%
f = figure();
for i = [1,3,5,7]
hold on;
t = 1:size(stats{i}, 1);
eps = stats{i}(:,1);

% plot(eps, t, 'LineWidth',1)
plot(eps, t)
end
set(gca, 'XScale', 'log');
%legend('k = 5','k = 10','k = 25','k = 50','k = 100', 'k = 250','k = 500')
legend('k = 5','k = 25','k = 100', 'k = 500')


ylabel 'iterations'
ax = get(f,'CurrentAxes');

set(ax,'XScale', 'log');
xlabel('$\bar{\epsilon}$', 'Interpreter','Latex')

xlim([1e-6, 1]);