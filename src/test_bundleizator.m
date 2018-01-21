% %% MONKS data import and labels fixing
% monk1_training_set = importdata("../data/monks-1.train.csv");
% monk1_test_set = importdata("../data/monks-1.test.clean.csv");
% monk1_train_classes = monk1_training_set(:,1) * 2 - 1;
% monk1_train_features = monk1_training_set(:,2:end);
% monk1_test_classes = monk1_test_set(:,1) * 2 - 1;
% monk1_test_features = monk1_test_set(:,2:end);
% 
% monk2_training_set = importdata("../data/monks-2.train.csv");
% monk2_test_set = importdata("../data/monks-2.test.clean.csv");
% monk2_train_classes = monk2_training_set(:,1) * 2 - 1;
% monk2_train_features = monk2_training_set(:,2:end);
% monk2_test_classes = monk2_test_set(:,1) * 2 - 1;
% monk2_test_features = monk2_test_set(:,2:end);
% 
% monk3_training_set = importdata("../data/monks-3.train.csv");
% monk3_test_set = importdata("../data/monks-3.test.clean.csv");
% monk3_train_classes = monk3_training_set(:,1) * 2 - 1;
% monk3_train_features = monk3_training_set(:,2:end);
% monk3_test_classes = monk3_test_set(:,1) * 2 - 1;
% monk3_test_features = monk3_test_set(:,2:end);
% 
%% MONKS tests

%kernel = @(x,y) (x*y')^2;
kernel = @(x,y) exp(-sum((x-y).^2));
%kernel = @(x,y) (x*y')^3;
C = logspace(-3, 3, 100);
bundleizator_precision = 1e-6;

% test_bundleizator_monk(monk1_train_features, monk1_train_classes, monk1_test_features, monk1_test_classes, C, kernel, bundleizator_precision);
% test_bundleizator_monk(monk2_train_features, monk2_train_classes, monk2_test_features, monk2_test_classes, C, kernel, bundleizator_precision);
% test_bundleizator_monk(monk3_train_features, monk3_train_classes, monk3_test_features, monk3_test_classes, C, kernel, bundleizator_precision);

%% ML-CUP tests
epsilon = 0:0.2:1;

% Data import
mlcup_training_set = importdata("../data/ml-cup17.train.csv");
mlcup_inputs = mlcup_training_set(:,2:end-2);
mlcup_outputs1 = mlcup_training_set(:,end-1);
mlcup_outputs2 = mlcup_training_set(:,end);

%split mlcup training set in training + test
train_proportion = 2 * ceil(length(mlcup_inputs)/3);
mlcup_train_inputs = mlcup_inputs(1:train_proportion,:);
mlcup_train_outputs1 = mlcup_outputs1(1:train_proportion,:);
mlcup_train_outputs2 = mlcup_outputs1(2:train_proportion,:);
mlcup_test_inputs = mlcup_inputs(train_proportion+1:end,:);
mlcup_test_outputs1 = mlcup_outputs1(train_proportion+1:end,:);
mlcup_test_outputs2 = mlcup_outputs2(train_proportion+1:end,:);

test_bundleizator_mlcup(mlcup_train_inputs, mlcup_train_outputs1, mlcup_test_inputs, mlcup_test_outputs1, C, epsilon, kernel, bundleizator_precision);
stopppp
test_bundleizator_mlcup(output2);