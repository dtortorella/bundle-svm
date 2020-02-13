addpath('../src/');

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

k = [5, 25, 100, 500];

%% non pruning values
tic;
[u_non_pruning,~,~,~,stat_non_pruning] = bundleizator(X, y1, C, kernel, ...
      @(f,Y) einsensitive_loss(f, Y, epsilon), @(f,Y) einsensitive_dloss(f, Y, epsilon),...
      precision);
time_non_pruning = toc; 

%%

for i = 1:4
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
f = figure();
for i = 1:4
hold on;
t = 1:size(stats{i}, 1);
eps = stats{i}(:,1);

plot(eps, t)
end

set(gca, 'XScale', 'log');
legend('k = 5','k = 25','k = 100', 'k = 500')
ylabel 'iterations'
ax = get(f,'CurrentAxes');
set(ax,'XScale', 'log');
xlabel('$\bar{\epsilon}$', 'Interpreter','Latex')
xlim([1e-6, 1]);