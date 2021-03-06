addpath('./src')
%% Load MLCUP17 dataset
mlcup_training_set = importdata("./data/ml-cup17.train.csv");
mlcup_train_outputs = mlcup_training_set(:,end-1:end);

X = mlcup_training_set(:,2:end-2);
y = mlcup_training_set(:,end-1);

%% Kernel, loss and params definition
kernel = @(x,y) exp(-.1*norm(x-y)^2);
kernel2 = @(x,y) (x*y').^2;

loss = @(f,y) einsensitive_loss(f, y, 0.2);
dloss = @(f,y) einsensitive_dloss(f, y, 0.2);

C = 1e4;
bar_eps = 1e-4;

max_inactive_count = 5;
inactive_zero_threshold = 1e-6;

%% Bundleizator run
tic;
[u, sv, t, eps] = bundleizator(X, y, C, kernel, loss, dloss , bar_eps);
t_bundle = toc;

%% Bundleizator run with pruning
tic;
[u_p, sv_p, t_p, eps_p] = bundleizator_pruning(X, y, C, kernel, loss, dloss, bar_eps, max_inactive_count, inactive_zero_threshold);
t_pruning = toc;

%% QP solver for original problem
tic;
[u_qp, sv_qp, J] = big_fat_solver(X, y, C, kernel, bar_eps, 'einsensitive', 0.2);
t_qp = toc;

%% 
fprintf("Times of execuion comparison \n");
fprintf("bundle method \t pruning \t qp original \n");
fprintf("%f \t %f \t %f \n", t_bundle, t_pruning, t_qp);


%%
%% Span vector selection

G = gram_matrix(X, kernel2);
Gn = gram_norm_matrix(X,kernel2);

f = 1;
tol = 1e-6;

sv_srrqr = select_span_vectors(G, 'sRRQR', f, tol);
sv_isvd = select_span_vectors(G, 'isvd', tol);
sv_iqr = select_span_vectors(G, 'iqr', tol);
sv_qr = select_span_vectors(G);

sv = sv_srrqr;

o = eval_orthonorm(sv_srrqr, Gn, 'normalize');

nonsv = 1:size(G,2);
nonsv(sv_srrqr) = [];
pmin = eval_parallelity(sv_srrqr, nonsv ,Gn, 'min');
pmean = eval_parallelity(sv_srrqr, nonsv, Gn, 'mean');

GX = G(:,sv_srrqr);
G = G(sv_srrqr,sv_srrqr);





