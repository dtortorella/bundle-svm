function [kernel, nu] = svm_select_model_bayesianly(features, classes, folds, kernels)
% SVM_SELECT_MODEL_BAYESIANLY Selects the hyperparameters of a n-SVM via cross-validation
%
% SYNOPSIS: [kernel, nu] = svm_select_model_bayesianly(features, classes, folds, kernels)
%
% INPUT:
% - features: a matrix containing one sample feature vector per row
% - classes: a column vector containing one sample class per entry, must be +/-1
% - folds: the number of folds for the cross-validation
% - kernels: a map of strings -> kernel functions
%
% OUTPUT:
% - kernel: the best kernel function selected
% - nu: the best value selected for this hyperparameter
%
% REMARKS This implementation uses bayesian optimization to find the best settings
%
% SEE ALSO svm_select_model, svr_select_model_bayesianly

    parameter_kernel = optimizableVariable('kernel', kernels.keys, 'Type', 'categorical');
    parameter_nu = optimizableVariable('nu', [0,1], 'Type', 'real');
    objective_function = @(x) crossvalidation_error(features, classes, folds, kernels, x);
    
    result = bayesopt(objective_function, [parameter_kernel, parameter_nu]);
    hyperparameters = table2struct(result.XAtMinObjective);
    
    kernel = hyperparameters.kernel;
    nu = hyperparameters.nu;
end

function inaccuracy = crossvalidation_error(features, classes, folds, kernels, hyperparameters)
% Computes the cross-validation error (1 - accuracy) for this dataset and hyperparameters
    hyperparameters = table2struct(hyperparameters);
    dataset_partition = kfolds_partition(size(features, 1), folds);
    validation_error = zeros(1, folds);  % keeps the validation MSE for each fold

    for fold_index = 1:folds
        % pick one fold at a time for validation, leave the others for training
        training_features = features((dataset_partition ~= fold_index),:);
        training_classes = classes((dataset_partition ~= fold_index),:);
        validation_features = features((dataset_partition == fold_index),:);
        validation_classes = classes((dataset_partition == fold_index),:);
        % train and estimate the validation accuracy for this partition
        model = svm_train(training_features, training_classes, kernels(char(hyperparameters.kernel)), hyperparameters.nu);
        if isstruct(model)
            predictions = svm_predict(model, validation_features);
            validation_error(fold_index) = sum(predictions ~= validation_classes) / length(validation_classes);
        else
            validation_error(fold_index) = 1;
        end
    end

    inaccuracy = mean(validation_error);
end