addpath('../src/');

% dataset
mlcup_training_set = importdata("../data/ml-cup17.train.csv");
mlcup_test_set = importdata("../data/ml-cup17.test.csv");
X = mlcup_training_set(:,2:end-2);
mlcup_train_outputs = mlcup_training_set(:,end-1:end);
y = mlcup_training_set(:,end-1);
kernel = @(x,y) exp(-.1*norm(x-y)^2);
% parameters to explore
C = 10.^(2:6);

eps = logspace(-8, -2, 13); % \bar{epsilon}

% mean of trials as reported time
trials = 3;


%% QP one iter
for i = 1:length(C)
    for j = 1:length(eps)
        for k = 1:trials
            tic;
            [u, sv, J] = big_fat_solver(X, y, C(i), kernel, eps(j), 'epsilon', 0.2);
            time(k) = toc;
        end
        Tqp(i,j) = mean(time);
    end
end


%% Bundleizator
k=5;
inac_thres = 1e-4;
for i = 1:length(C)
    for j = 1:length(eps)
        for k = 1:trials
            tic;
            [u, sv] = bundleizator_pruning(X, y, C(i), kernel, @(f,y) einsensitive_loss(f, y, 0.2), @(f,y) einsensitive_dloss(f, y, 0.2), eps(j), k, inac_thres);
            %[u, sv] = bundleizator(X, y, C(i), kernel, @(f,y) einsensitive_loss(f, y, 0.2), @(f,y) einsensitive_dloss(f, y, 0.2), eps(j));
            time(k) = toc;
        end
        Tbzt(i,j) = mean(time);
    end
end
