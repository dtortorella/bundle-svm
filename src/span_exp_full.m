%% ionosphere
load 'ionosphere'

kernel = @(x,y) (x*y').^2;

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

figure_ort_par('QR, ionosphere, normalized quadratic gram', S, ort_norm, par_norm, parm_norm);

%% ionophere, non-normalized

G = gram_matrix(X, kernel);
[Q,R,p] = qr(G,0);
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

figure_ort_par('QR, ionosphere, quadratic gram', S, ort, par, parm);
%% SVD
% [U,R,~] = svd(Gn);
% p = 1:size(X,1);

% kernel = @(x,y) exp(-norm(x-y)^2);
% ker_str = 'exponential'


%%

% load 'ovariancancer'
% X = obs;

% load 'fisheriris'
% X = meas;
% mlcup_training_set = importdata("../data/ml-cup17.train.csv");
% X = mlcup_training_set(:,2:end-2);


