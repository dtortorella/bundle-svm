%% ionosphere
load 'ionosphere'

kernel = @(x,y) (x*y').^2;

Gn = gram_norm_matrix(X, kernel);

[Q,R,p] = qr(Gn,0);
maxs = min(-1,floor(log10(max(abs(diag(R))))));

par_norm = [];
ort_norm = [];
parm_norm = [];
    
tol = logspace(maxs-8,maxs,60);
for i = 1:size(tol,2) 
    
    [Q,R,p,~] = online_qr(Gn,tol(i));
    S = abs(diag(R));

    % [U,R,~] = svd(Gn);
    % p = 1:size(X,1);
    
    np = 1:size(Gn,1);
    np(p) = [];
    par_norm(i) = eval_parallelity(p, np, Gn,'min');
    parm_norm(i) = eval_parallelity(p, np, Gn,'mean');
    ort_norm(i) = eval_orthonorm(p, Gn, 'normalize');
end
figure_ort_par_online('oQR, ionosphere, normalized quadratic gram',tol, ort_norm, par_norm, parm_norm);

%% ionophere, non-normalized

G = gram_matrix(X, kernel);

[Q,R,p] = qr(G,0);
maxs = min(-1,floor(log10(max(abs(diag(R))))));

par = [];
ort = [];
parm = [];
    
tol = logspace(maxs-8,maxs,60);
for i = 1:size(tol,2) 
    
    [Q,R,p,~] = online_qr(G,tol(i));
    S = abs(diag(R));
    % [U,R,~] = svd(Gn);
    % p = 1:size(X,1);
    
    np = 1:size(G,1);
    np(p) = [];
    par(i) = eval_parallelity(X(p,:), X(np,:), kernel,'min');
    parm(i) = eval_parallelity(X(p,:), X(np,:), kernel,'mean');   
    ort(i) = eval_orthonorm(X(p,:), kernel, 'normalize');

end
figure_ort_par_online('oQR, ionosphere, quadratic gram',tol, ort, par, parm);

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


