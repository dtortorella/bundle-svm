addpath('./src')
%% Monks experiments
%monks_kernels = containers.Map({'linear', 'quadratic', 'cubic', 'gaussian'}, {'-t 0', '-t 1 -d 2', '-t 1 -d 3',  '-t 2'});
monks_kernels_libsvm = containers.Map({'linear homog.', 'linear (c=0.5)', 'linear (c=1)', 'linear (c=5)', ...
 'quadratic homog.', 'quadratic (c=0.5)', 'quadratic (c=1)', 'quadratic (c=5)', ...
 'cubic homog.', 'cubic (c=0.5)', 'cubic (c=1)', 'cubic (c=5)', ...
 'gaussian (gamma=0.1)', 'gaussian (gamma=0.5)', 'gaussian (gamma=1)', 'gaussian (gamma=5)'}, ...
{'-t 0', '-t 0 -r 0.5', '-t 0 -r 1', '-t 0 -r 5', ...
 '-t 1 -d 2', '-t 1 -d 2 -r 0.5', '-t 1 -d 2 -r 1', '-t 1 -d 2 -r 5', ...
 '-t 1 -d 3', '-t 1 -d 3 -r 0.5', '-t 1 -d 3 -r 1', '-t 1 -d 3 -r 5', ...
 '-t 2 -g 0.1', '-t 2 -g 0.5', '-t 2 -g 1', '-t 2 -g 5'});
monks_kernels = containers.Map({'linear homog.', 'linear (c=0.5)', 'linear (c=1)', 'linear (c=5)', ...
 'quadratic homog.', 'quadratic (c=0.5)', 'quadratic (c=1)', 'quadratic (c=5)', ...
 'cubic homog.', 'cubic (c=0.5)', 'cubic (c=1)', 'cubic (c=5)', ...
 'gaussian (gamma=0.1)', 'gaussian (gamma=0.5)', 'gaussian (gamma=1)', 'gaussian (gamma=5)'}, ...
{@(x,y) (x*y'), @(x,y) (x*y' + .5), @(x,y) (x*y' + 1), @(x,y) (x*y' + 5), ...
 @(x,y) (x*y')^2, @(x,y) (x*y' + .5)^2, @(x,y) (x*y' + 1)^2, @(x,y) (x*y' + 5)^2, ...
 @(x,y) (x*y')^3, @(x,y) (x*y' + .5)^3, @(x,y) (x*y' + 1)^3, @(x,y) (x*y' + 5)^3, ...
 @(x,y) exp(-.1*norm(x-y)^2), @(x,y) exp(-.5*norm(x-y)^2), @(x,y) exp(-norm(x-y)^2), @(x,y) exp(-5*norm(x-y)^2)});


%% Monk 1 dataset experiment
 
% % Data import and labels fixing
monk1_training_set = importdata("./data/monks-1.train.csv");
monk1_test_set = importdata("./data/monks-1.test.clean.csv");

monk1_train_classes = monk1_training_set(:,1) * 2 - 1;
monk1_train_features = monk1_training_set(:,2:end);

monk1_test_classes = monk1_test_set(:,1) * 2 - 1;
monk1_test_features = monk1_test_set(:,2:end);
% 
% % Model selection
% monk1_folds = 5;
% monk1_nu_range = 0.1:0.1:min(max_feasible_nu(monk1_train_classes, monk1_folds));
% check_balance(monk1_train_classes, monk1_folds);
% 
% pause
% 
% [monk1_kernel_bayes, monk1_nu_bayes] = svm_select_model_bayesianly(monk1_train_features, monk1_train_classes, monk1_folds, monks_kernels);
% [monk1_kernel, monk1_nu] = svm_select_model(monk1_train_features, monk1_train_classes, monk1_folds, monks_kernels, monk1_nu_range);
% 
% fprintf("bayesian search \t| grid search \n ker: %d, nu: %f \t| ker: %d nu: %f\n", monk1_kernel_bayes, monk1_nu_bayes, monk1_kernel, monk1_nu);
% 
% % Training
% monk1_model = svm_train(monk1_train_features, monk1_train_classes, monk1_kernel , monk1_nu);
% monk1_model_bayes = svm_train(monk1_train_features, monk1_train_classes, monk1_kernel_bayes, monk1_nu_bayes);
% 
% % Test
% monk1_predictions = svm_predict(monk1_model, monk1_test_features);
% monk1_predictions_bayes = svm_predict(monk1_model_bayes, monk1_test_features);
% monk1_model_accurancy = sum(monk1_test_classes == monk1_predictions) / length(monk1_test_classes)
% monk1_model_bayes_accurancy = sum(monk1_test_classes == monk1_predictions_bayes) / length(monk1_test_classes)
% 
% pause

%% Monk 2 dataset experiment

% Data import and labels fixing
monk2_training_set = importdata("./data/monks-2.train.csv");
monk2_test_set = importdata("./data/monks-2.test.clean.csv");

monk2_train_classes = monk2_training_set(:,1) * 2 - 1;
monk2_train_features = monk2_training_set(:,2:end);

monk2_test_classes = monk2_test_set(:,1) * 2 - 1;
monk2_test_features = monk2_test_set(:,2:end);

% Model Selection

% monk2_folds = 5;
% monk2_nu_range = 0.1:0.01:min(svm_max_feasible_nu(monk2_train_classes, monk2_folds));
% check_balance(monk2_train_classes, monk2_folds);
% 
% 
% [monk2_kernel, monk2_nu] = svm_select_model(monk2_train_features, monk2_train_classes, monk2_folds, monks_kernels, monk2_nu_range)%;
% 
% %[monk2_kernel_bayes, monk2_nu_bayes] = svm_select_model_bayesianly(monk2_train_features, monk2_train_classes, monk2_folds, monks_kernels);
% stopp
% 
% fprintf("bayesian search \t| grid search \n ker: %d, nu: %f \t| ker: %d nu: %f\n", monk2_kernel_bayes, monk2_nu_bayes, monk2_kernel, monk2_nu);
% 
% % Training 
% monk2_model = svm_train(monk2_train_features, monk2_train_classes, monk2_kernel, monk2_nu);
% monk2_model_bayes = svm_train(monk2_train_features, monk2_train_classes, monk2_kernel_bayes, monk2_nu_bayes);
% 
% % Test
% monk2_predictions = svm_predict(monk2_model, monk2_test_features);
% monk2_predictions_bayes = svm_predict(monk2_model_bayes, monk2_test_features);
% monk2_model_accurancy = sum(monk2_test_classes == monk2_predictions) / length(monk2_test_classes)
% monk2_model_bayes_accurancy = sum(monk2_test_classes == monk2_predictions_bayes) / length(monk2_test_classes)
% 
% pause

%% Monk 3 dataset experiment

% Data import and labels fixing
monk3_training_set = importdata("./data/monks-3.train.csv");
monk3_test_set = importdata("./data/monks-3.test.clean.csv");
 
monk3_train_classes = monk3_training_set(:,1) * 2 - 1;
monk3_train_features = monk3_training_set(:,2:end);

monk3_test_classes = monk3_test_set(:,1) * 2 - 1;
monk3_test_features = monk3_test_set(:,2:end);
% 
% % Model selection
% monk3_nu_range = 0.1:0.1:min(max_feasible_nu(monk3_train_classes, 5));
% [monk2_kernel, monk2_nu] = svm_select_model(monk2_train_features, monk2_train_classes, 5, {0,2}, monk3_nu_range);
% 
% % Training
% monk3_model = svm_train(monk3_train_features, monk3_train_classes, monk3_kernel , monk3_nu); 
% 
% % Test
% monk3_predictions = svm_predict(monk3_model, monk3_test_features);
% monk3_model_accurancy = sum(monk3_test_classes == monk3_predictions) / length(monk3_test_classes)

%% MLCUP17 dataset experiment
%Ml-cup kernels
mlcup_kernels_libsvm = containers.Map({'linear homog.', 'linear (c=0.5)', 'linear (c=1)', 'linear (c=5)', ...
 'quadratic homog.', 'quadratic (c=0.5)', 'quadratic (c=1)', 'quadratic (c=5)', ...
 'cubic homog.', 'cubic (c=0.5)', 'cubic (c=1)', 'cubic (c=5)', ...
 'gaussian (gamma=0.1)', 'gaussian (gamma=0.5)', 'gaussian (gamma=1)', 'gaussian (gamma=5)'}, ...
{'-t 0', '-t 0 -r 0.5', '-t 0 -r 1', '-t 0 -r 5', ...
 '-t 1 -d 2', '-t 1 -d 2 -r 0.5', '-t 1 -d 2 -r 1', '-t 1 -d 2 -r 5', ...
 '-t 1 -d 3', '-t 1 -d 3 -r 0.5', '-t 1 -d 3 -r 1', '-t 1 -d 3 -r 5', ...
 '-t 2 -g 0.1', '-t 2 -g 0.5', '-t 2 -g 1', '-t 2 -g 5'});
mlcup_kernels = containers.Map({'linear homog.', 'linear (c=0.5)', 'linear (c=1)', 'linear (c=5)', ...
 'quadratic homog.', 'quadratic (c=0.5)', 'quadratic (c=1)', 'quadratic (c=5)', ...
 'cubic homog.', 'cubic (c=0.5)', 'cubic (c=1)', 'cubic (c=5)', ...
 'gaussian (gamma=0.1)', 'gaussian (gamma=0.5)', 'gaussian (gamma=1)', 'gaussian (gamma=5)'}, ...
{@(x,y) (x*y'), @(x,y) (x*y' + .5), @(x,y) (x*y' + 1), @(x,y) (x*y' + 5), ...
 @(x,y) (x*y')^2, @(x,y) (x*y' + .5)^2, @(x,y) (x*y' + 1)^2, @(x,y) (x*y' + 5)^2, ...
 @(x,y) (x*y')^3, @(x,y) (x*y' + .5)^3, @(x,y) (x*y' + 1)^3, @(x,y) (x*y' + 5)^3, ...
 @(x,y) exp(-.1*norm(x-y)^2), @(x,y) exp(-.5*norm(x-y)^2), @(x,y) exp(-norm(x-y)^2), @(x,y) exp(-5*norm(x-y)^2)});

% Data import
mlcup_training_set = importdata("./data/ml-cup17.train.csv");
mlcup_test_set = importdata("./data/ml-cup17.test.csv");

mlcup_train_inputs = mlcup_training_set(:,2:end-2);
mlcup_train_outputs = mlcup_training_set(:,end-1:end);
mlcup_train_outputs1 = mlcup_training_set(:,end-1);
mlcup_train_outputs2 = mlcup_training_set(:,end);

mlcup_test_ids = mlcup_test_set(:,1);
mlcup_test_inputs = mlcup_test_set(:,2:end);

% Model selection 
% mlcup_nus = [0.1:0.1:1];
% mlcup_Cs = logspace(-3,3,7);
% kernel_container = containers.Map({'linear', 'cubic', 'gaussian'}, {0,1,2});
% 
% [mlcup_kernel1_bayes, mlcup_nu1_bayes, mlcup_C1_bayes] = svr_select_model_bayesianly(mlcup_train_inputs, mlcup_train_outputs1, 5, kernel_container);
% 
% pause
% 
% [mlcup_kernel1, mlcup_nu1, mlcup_C1, mean, devi] = svr_select_model(mlcup_train_inputs, mlcup_train_outputs1, 5, {1,2}, mlcup_nus, mlcup_Cs);
% 
% fprintf("bayesian search \t| grid search \n ker: %d, nu: %f, C:%f \t| ker: %d nu: %f C:%f\n", monk2_kernel_bayes, monk2_nu_bayes, monk2_kernel, monk2_nu);
% 
% 
% stop
% [mlcup_kernel2, mlcup_nu2, mlcup_C2] = svr_select_model(mlcup_train_inputs, mlcup_train_outputs2, 5, {0,1,2});
%  
% % Training models
% [mlcup_model1, mlcup_epsilon1] = svr_train(mlcup_train_inputs, mlcup_train_outputs1, mlcup_kernel1, mlcup_nu1, mlcup_C1);
% [mlcup_model2, mlcup_epsilon2] = svr_train(mlcup_train_inputs, mlcup_train_outputs2, mlcup_kernel2, mlcup_nu2, mlcup_C2);
% 
% % Predictions
% mlcup_predictions1 = svr_predict(mlcup_model1, mlcup_test_inputs);
% mlcup_predictions2 = svr_predict(mlcup_model2, mlcup_test_inputs);