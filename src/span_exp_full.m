% load 'ionosphere'
% dataset_s = "ionoshpere";

mlcup_training_set = importdata("../data/ml-cup17.train.csv");
X = mlcup_training_set(:,2:end-2);
dataset_s = "MLCUP";

% kernel = @(x,y) exp(-norm(x-y)^2);
% ker_s = "exponential";

kernel = @(x,y) (x*y').^2;
ker_s = "quadratic";

%% QR

Gn = gram_norm_matrix(X, kernel);

[Q,R,p] = qr(Gn,0);
S = abs(diag(R));

par_norm = [];
ort_norm = [];
parm_norm = [];

for i = 1:size(X,1)
   if i ~= size(X,1)
       par_norm(i) = eval_parallelity(p(1:i), p(i+1:end), Gn,'min');
       parm_norm(i) = eval_parallelity(p(1:i), p(i+1:end), Gn,'mean');
   end
   ort_norm(i) = eval_orthonorm(p(1:i), Gn, 'normalize');
end

figure_ort_par('QR on $\hat{G}$', S, ort_norm, par_norm, parm_norm);
xlim([1 1013])

%% QR non-normalized

G = gram_matrix(X, kernel);
[~,R,p] = qr(G,0);
S = abs(diag(R));

ort = [];
par = [];
parm = [];

for i = 1:size(X,1)
   if i ~= size(X,1)
       par(i) = eval_parallelity(X(p(1:i),:), X(p(i+1:end),:), kernel,'min');
       parm(i) = eval_parallelity(X(p(1:i),:), X(p(i+1:end),:), kernel,'mean');
   end
   ort(i) = eval_orthonorm(X(p(1:i),:), kernel, 'normalize');
end

figure_ort_par("QR on G", S, ort, par, parm);
%% SVD

Gn = gram_norm_matrix(X, kernel);

[~,S,~] = svd(Gn,0);

p = 1:size(X,1);

par_norm = [];
ort_norm = [];
parm_norm = [];

for i = 1:size(X,1)
   if i ~= size(X,1)
       par_norm(i) = eval_parallelity(p(1:i), p(i+1:end), Gn,'min');
       parm_norm(i) = eval_parallelity(p(1:i), p(i+1:end), Gn,'mean');
   end
   ort_norm(i) = eval_orthonorm(p(1:i), Gn, 'normalize');
end

s = sprintf("SVD, %s, normalized %s gram", dataset_s, ker_s);
figure_ort_par(s, diag(S), ort_norm, par_norm, parm_norm);

% non-normalized

G = gram_matrix(X, kernel);
[~,S,~] = svd(G);
p = 1:size(X,1);

ort = [];
par = [];
parm = [];

for i = 1:size(X,1)
   if i ~= size(X,1)
       par(i) = eval_parallelity(X(p(1:i),:), X(p(i+1:end),:), kernel,'min');
       parm(i) = eval_parallelity(X(p(1:i),:), X(p(i+1:end),:), kernel,'mean');
   end
   ort(i) = eval_orthonorm(X(p(1:i),:), kernel, 'normalize');
end

s = sprintf("SVD, %s, %s gram", dataset_s, ker_s);
figure_ort_par(s, diag(S), ort, par, parm);

