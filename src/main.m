%% Monk 1 dataset experiment

% Data import and normalization
monk1_trainingset = importdata("../data/monks-1.train.csv");
monk1_testset = importdata("../data/monks-1.test.csv");

monk1_TR_classes = monk1_trainingset(:,1)*2 - ones(length(monk1_trainingset),1);
monk1_TR_features = monk1_trainingset(:,2:end);

monk1_TS_classes = monk1_testset(:,1)*2 - ones(length(monk1_testset),1);
monk1_TS_features = monk1_testset(:,2:end);

% Model selection 
%
%
monk1_kernel = 2; %using rbf kernel (see svm_train file)
monk1_nu = 0.5;
%

% Training
monk1_model = svm_train(monk1_TR_features, monk1_TR_classes, monk1_kernel , monk1_nu); 

% Test
monk1_predictions = svm_predict(monk1_model, monk1_TS_features)
monk1_model_accurancy = 1 - nnz(monk1_TS_classes - monk1_predictions)/length(monk1_testset)

%% Monk 2 dataset experiment

% Data import and normalization
monk2_trainingset = importdata("../data/monks-2.train.csv");
monk2_testset = importdata("../data/monks-2.test.csv");

monk2_TR_classes = monk2_trainingset(:,1)*2 - ones(length(monk2_trainingset),1);
monk2_TR_features = monk2_trainingset(:,2:end);

monk2_TS_classes = monk2_testset(:,1)*2 - ones(length(monk2_testset),1);
monk2_TS_features = monk2_testset(:,2:end);

% Model selection 
%
%
monk2_kernel = 2; %using rbf kernel (see svm_train file)
monk2_nu = 0.5;
%

% Training
monk2_model = svm_train(monk2_TR_features, monk2_TR_classes, monk2_kernel , monk2_nu); 

% Test
monk2_predictions = svm_predict(monk2_model, monk2_TS_features);
monk2_model_accurancy = 1 - nnz(monk2_TS_classes - monk2_predictions)/length(monk2_testset)


%% Monk 3 dataset experiment

% Data import and normalization
monk3_trainingset = importdata("../data/monks-3.train.csv");
monk3_testset = importdata("../data/monks-3.test.csv");

monk3_TR_classes = monk3_trainingset(:,1)*2 - ones(length(monk3_trainingset),1);
monk3_TR_features = monk3_trainingset(:,2:end);

monk3_TS_classes = monk3_testset(:,1)*2 - ones(length(monk3_testset),1);
monk3_TS_features = monk3_testset(:,2:end);

% Model selection 
%
%
monk3_kernel = 2; %using rbf kernel (see svm_train file)
monk3_nu = 0.5;
%

% Training
monk3_model = svm_train(monk3_TR_features, monk3_TR_classes, monk3_kernel , monk3_nu); 

% Test
monk3_predictions = svm_predict(monk3_model, monk3_TS_features);
monk3_model_accurancy = 1 - nnz(monk3_TS_classes - monk3_predictions)/length(monk3_testset)


%% MLCUP17 dataset experiment

% Data import and normalization
mlcup_trainingset = importdata("../data/ml-cup17.train.csv");
mlcup_testset = importdata("../data/ml-cup17.test.csv");

mlcup_TR_ids = mlcup_trainingset(:,1);
mlcup_TR_inputs = mlcup_trainingset(:,2:end-2);
mlcup_TR_outputs1 = mlcup_trainingset(:,end-1);
mlcup_TR_outputs2 = mlcup_trainingset(:,end);
mlcup_TS_ids = mlcup_testset(:,1);
mlcup_TS_inputs = mlcup_testset(:,2:end);


% Model selection 
%
%
mlcup_nu1 = 0.5;
mlcup_kernel1 = 2; %using rbf kernel (see svm_train file)
mlcup_C1 = iqr(mlcup_TR_outputs1)/1.349; % C value computed as the standard value in BoxConstraint of fitrsvm with rbf kernel

mlcup_nu2 = 0.5;
mlcup_kernel2 = 2;
mlcup_C2 = iqr(mlcup_TR_outputs2)/1.349;
%


% Training models
[mlcup_model1, mlcup_epsilon1] = svr_train(mlcup_TR_inputs, mlcup_TR_outputs1, mlcup_kernel1, mlcup_nu1, mlcup_C1);
[mlcup_model2, mlcup_epsilon2] = svr_train(mlcup_TR_inputs, mlcup_TR_outputs2, mlcup_kernel2, mlcup_nu2, mlcup_C2);

% Predictions
mlcup_predictions1 = svr_predict(mlcup_model1, mlcup_TS_inputs);
mlcup_predictions2 = svr_predict(mlcup_model2, mlcup_TS_inputs);