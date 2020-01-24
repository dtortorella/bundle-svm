ker_poly = @(x,y) (x*y' + 1)^2;
ker_exp = @(x,y) exp(-.5*norm(x-y)^2);

kernels_libsvm = containers.Map({'quadratic homog.', 'quadratic (c=5)', ...
 'gaussian (gamma=0.5)', 'gaussian (gamma=1)'}, ...
{'-t 1 -d 2 -r 0', '-t 1 -d 2 -r 5', ...
 '-t 2 -g 0.5', '-t 2 -g 5'});

kernels = containers.Map({
 'quadratic homog.', 'quadratic (c=5)', ...
 'gaussian (gamma=0.5)', 'gaussian (gamma=5)'}, ...
{
 @(x,y) (x*y')^2, @(x,y) (x*y' + 5)^2, ...
 @(x,y) exp(-.5*norm(x-y)^2), @(x,y) exp(-5*norm(x-y)^2)});

mlcup_training_set = importdata("../data/ml-cup17.train.csv");
X = mlcup_training_set(:,2:end-2);

%load 'ionosphere'
%X = X;

%load 'fisheriris'
%X = meas;

%load 'ovariancancer'
%X = obs;

%monk2
%X = importdata("../data/monks-3.train.csv");
%X = X(:,2:end);


X = X(randperm(size(X, 1)), :);

kernel_index = kernels.keys;
kernel = kernels(kernel_index{3});
k_str = kernel_index{3};

G = gram_matrix(X,kernel);

algs = ["qr","online_qr","online_svd"];
tols = [10e-5, 10e-6,10e-7, 10e-8, 10e-9, 10e-10];


fprintf('algorithm | value tol    | kernel  |  compu.time  | #sv |SV_rnk|G_rank| o_norm\n'); 

for alg = algs
    for tol = tols
        for i = 1
        
        tic;
        sv = select_span_vectors(G, alg, tol);
        t = toc;

        est_rank = rank(G(sv,sv));
        real_rank = rank(G);

        o = eval_orthonorm(X(sv,:), kernel, 'normalize');

        fprintf('%s | %e | %s | %d | %d | %d  | %d | %e\n', alg, tol, k_str, t, size(sv,2), est_rank, real_rank, o);
        end
    end
end



