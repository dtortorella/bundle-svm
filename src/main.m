%% Monk 1 dataset experiment

% Data import and labels fixing
monk1_training_set = importdata("../data/monks-1.train.csv");
monk1_test_set = importdata("../data/monks-1.test.csv");

monk1_train_classes = monk1_training_set(:,1) * 2 - 1;
monk1_train_features = monk1_training_set(:,2:end);

monk1_test_classes = monk1_test_set(:,1) * 2 - 1;
monk1_test_features = monk1_test_set(:,2:end);

% Model selection
[monk1_kernel, monk1_nu] = svm_select_model(monk1_train_features, monk1_train_classes, 5, {0,2});

% Training
monk1_model = svm_train(monk1_train_features, monk1_train_classes, monk1_kernel , monk1_nu); 

% Test
monk1_predictions = svm_predict(monk1_model, monk1_test_features);
monk1_model_accurancy = sum(monk1_test_classes == monk1_predictions) / length(monk1_test_classes)


%% Monk 2 dataset experiment

% Data import and labels fixing
monk2_training_set = importdata("../data/monks-2.train.csv");
monk2_test_set = importdata("../data/monks-2.test.csv");

monk2_train_classes = monk2_training_set(:,1) * 2 - 1;
monk2_train_features = monk2_training_set(:,2:end);

monk2_test_classes = monk2_test_set(:,1) * 2 - 1;
monk2_test_features = monk2_test_set(:,2:end);

% Model selection
[monk2_kernel, monk2_nu] = svm_select_model(monk2_train_features, monk2_train_classes, 5, {0,2});

% Training
monk2_model = svm_train(monk2_train_features, monk2_train_classes, monk2_kernel , monk2_nu); 

% Test
monk2_predictions = svm_predict(monk2_model, monk2_test_features);
monk2_model_accurancy = sum(monk2_test_classes == monk2_predictions) / length(monk2_test_classes)

%% Monk 3 dataset experiment

% Data import and labels fixing
monk3_training_set = importdata("../data/monks-3.train.csv");
monk3_test_set = importdata("../data/monks-3.test.csv");

monk3_train_classes = monk3_training_set(:,1) * 2 - 1;
monk3_train_features = monk3_training_set(:,2:end);

monk3_test_classes = monk3_test_set(:,1) * 2 - 1;
monk3_test_features = monk3_test_set(:,2:end);

% Model selection
[monk2_kernel, monk2_nu] = svm_select_model(monk2_train_features, monk2_train_classes, 5, {0,2});

% Training
monk3_model = svm_train(monk3_train_features, monk3_train_classes, monk3_kernel , monk3_nu); 

% Test
monk3_predictions = svm_predict(monk3_model, monk3_test_features);
monk3_model_accurancy = sum(monk3_test_classes == monk3_predictions) / length(monk3_test_classes)

%% MLCUP17 dataset experiment

% Data import
mlcup_training_set = importdata("../data/ml-cup17.train.csv");
mlcup_test_set = importdata("../data/ml-cup17.test.csv");

mlcup_train_inputs = mlcup_training_set(:,2:end-2);
mlcup_train_outputs1 = mlcup_training_set(:,end-1);
mlcup_train_outputs2 = mlcup_training_set(:,end);

mlcup_test_ids = mlcup_test_set(:,1);
mlcup_test_inputs = mlcup_test_set(:,2:end);


% Model selection 
%
%
mlcup_nu1 = 0.5;
mlcup_kernel1 = 2; %using rbf kernel (see svm_train file)
mlcup_C1 = iqr(mlcup_train_outputs1)/1.349; % C value computed as the standard value in BoxConstraint of fitrsvm with rbf kernel

mlcup_nu2 = 0.5;
mlcup_kernel2 = 2;
mlcup_C2 = iqr(mlcup_train_outputs2)/1.349;
%


% Training models
[mlcup_model1, mlcup_epsilon1] = svr_train(mlcup_train_inputs, mlcup_train_outputs1, mlcup_kernel1, mlcup_nu1, mlcup_C1);
[mlcup_model2, mlcup_epsilon2] = svr_train(mlcup_train_inputs, mlcup_train_outputs2, mlcup_kernel2, mlcup_nu2, mlcup_C2);

% Predictions
mlcup_predictions1 = svr_predict(mlcup_model1, mlcup_test_inputs);
mlcup_predictions2 = svr_predict(mlcup_model2, mlcup_test_inputs);
