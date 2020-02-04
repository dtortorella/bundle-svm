% load 'ionosphere'
% dataset_s = "ionoshpere";

mlcup_training_set = importdata("../data/ml-cup17.train.csv");
X = mlcup_training_set(:,2:end-2);
dataset_s = "MLCUP";

% kernel = @(x,y) exp(-norm(x-y)^2);
% ker_s = "exponential";

kernel = @(x,y) (x*y').^2;
ker_s = "quadratic";

%% oQR

Gn = gram_norm_matrix(X, kernel);

[~,R,~] = qr(Gn,0);
maxs = min(0,floor(log10(max(abs(diag(R))))));

par_norm = [];
ort_norm = [];
parm_norm = [];
nsv_norm = [];

tol = logspace(maxs-4,maxs,60);
for i = 1:size(tol,2) 
    
    [Q,R,p,~] = online_qr(Gn,tol(i));
    S = abs(diag(R));

    np = 1:size(Gn,1);
    np(p) = []; % np are indexes not in p
    nsv_norm(i) = size(p,2);
    par_norm(i) = eval_parallelity(p, np, Gn,'min');
    parm_norm(i) = eval_parallelity(p, np, Gn,'mean');
    ort_norm(i) = eval_orthonorm(p, Gn, 'normalize');
end

figure_ort_par_online("Incremental on $\hat{G}$",tol, nsv_norm, ort_norm, par_norm, parm_norm);


%% non-normalized

G = gram_matrix(X, kernel);

[~,R,~] = qr(G,0);
maxs = min(2,floor(log10(max(abs(diag(R))))));

par = [];
ort = [];
parm = [];
nsv = [];
tol = logspace(maxs-4,maxs,60);

for i = 1:size(tol,2) 
    
    [Q,R,p,sv] = online_qr(G,tol(i));
    S = abs(diag(R));
    
    nsv(i) = size(p,2);
    np = 1:size(G,1);
    np(p) = [];
    par(i) = eval_parallelity(p, np, Gn,'min');
    parm(i) = eval_parallelity(p, np, Gn,'mean');
    ort(i) = eval_orthonorm(p, Gn, 'normalize');

end
s = "Incremental on $G$";
figure_ort_par_online(s,tol, nsv, ort, par, parm);


%% oSVD

Gn = gram_norm_matrix(X, kernel);

[~,S,~] = svd(Gn,0);
maxs = min(-1,floor(log10(max(abs(diag(S))))));

par_norm = [];
ort_norm = [];
parm_norm = [];
nsv_norm = [];

tol = logspace(maxs-8,maxs,60);
for i = 1:size(tol,2) 
    
    [~,S,p] = online_svd(Gn);
    
    np = 1:size(Gn,1);
    np(p) = []; % np are indexes not in p
    nsv_norm(i) = size(p,2);
    par_norm(i) = eval_parallelity(p, np, Gn,'min');
    parm_norm(i) = eval_parallelity(p, np, Gn,'mean');
    ort_norm(i) = eval_orthonorm(p, Gn, 'normalize');
end
s = sprintf("oSVD, %s, normalized %s gram", dataset_s, ker_s);
figure_ort_par_online(s,tol, nsv_norm, ort_norm, par_norm, parm_norm);

% non-normalized

G = gram_matrix(X, kernel);

[~,S,~] = svd(G,0);
maxs = min(-1,floor(log10(max(abs(diag(S))))));

par = [];
ort = [];
parm = [];
nsv = [];
tol = logspace(maxs-8,maxs,60);
for i = 1:size(tol,2) 
    
    [~,S,p] = online_svd(G);
    
    nsv(i) = size(p,2);
    np = 1:size(G,1);
    np(p) = [];
    par(i) = eval_parallelity(X(p,:), X(np,:), kernel,'min');
    parm(i) = eval_parallelity(X(p,:), X(np,:), kernel,'mean');   
    ort(i) = eval_orthonorm(X(p,:), kernel, 'normalize');

end
s = sprintf("oSVD, %s, %s gram", dataset_s, ker_s);
figure_ort_par_online(s,tol, nsv, ort, par, parm);

%% sRRQR

Gn = gram_norm_matrix(X, kernel);

[~,R,~] = qr(Gn,0);
maxs = min(0,floor(log10(max(abs(diag(R))))));

fs = [1,2,5,10];
for f = fs
    par_norm = [];
    ort_norm = [];
    parm_norm = [];
    nsv_norm = [];

    tol = logspace(maxs-4,maxs,60);
    for i = 1:size(tol,2) 

        [Q,R,p] = sRRQR_tol(Gn,f,tol(i));
        S = abs(diag(R));
        p = p(1:size(Q,2));
        np = 1:size(Gn,1);
        np(p) = []; % np are indexes not in p
        nsv_norm(i) = size(p,2);
        par_norm(i) = eval_parallelity(p, np, Gn,'min');
        parm_norm(i) = eval_parallelity(p, np, Gn,'mean');
        ort_norm(i) = eval_orthonorm(p, Gn, 'normalize');
    end
    s = sprintf("sRRQR on $\\hat{G}$ , f = %d",f);
    figure_ort_par_online(s ,tol, nsv_norm, ort_norm, par_norm, parm_norm);
    drawnow
    % non-normalized

%     G = gram_matrix(X, kernel);

% 
%     [~,R,~] = qr(G,0);
%     maxs = min(0,floor(log10(max(abs(diag(R))))));
% 
%     par = [];
%     ort = [];
%     parm = [];
%     nsv = [];
%     tol = logspace(maxs-5,maxs,60);
% 
%     for i = 1:size(tol,2) 
% 
%         [Q,R,p] = sRRQR_tol(G,f,tol(i));
%         S = abs(diag(R));
%         p = p(1:size(Q,2));
%         
%         nsv(i) = size(p,2);
%         np = 1:size(G,1);
%         np(p) = [];
%         par(i) = eval_parallelity(p, np, Gn,'min');
%         parm(i) = eval_parallelity(p, np, Gn,'mean');
%         ort(i) = eval_orthonorm(p, Gn, 'normalize');
% 
%     end
%     s = sprintf("sRRQR on $G$ , f = %d",f);
%     figure_ort_par_online(s,tol, nsv, ort, par, parm);
%     drawnow
end
