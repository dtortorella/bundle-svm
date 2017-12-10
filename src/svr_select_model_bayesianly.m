function [kernel, nu, C] = svr_select_model_bayesianly(inputs, outputs, folds, kernels)
% SVR_SELECT_MODEL_BAYESIANLY Selects the hyperparameters of a n-SVR via cross-validation
%
% SYNOPSIS: [kernel, nu, C] = svr_select_model_bayesianly(inputs, outputs, folds, kernels)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a column vector containing one output sample per entry
% - folds: the number of folds for the cross-validation
% - kernels: a map of strings -> kernel functions
%
% OUTPUT:
% - kernel: the best kernel function selected
% - nu: the best value selected for this hyperparameter
% - C: the best value selected for this hyperparameter
%
% REMARKS This implementation uses bayesian optimization to find the best settings
%
% SEE ALSO svr_select_model, svm_select_model_bayesianly

    parameter_kernel = optimizableVariable('kernel', kernels.keys, 'Type', 'categorical');
    parameter_nu = optimizableVariable('nu', [0,1], 'Type', 'real');
    parameter_C = optimizableVariable('C', [1e-3,1e+5], 'Type', 'real');
    objective_function = @(x) crossvalidation_error(inputs, outputs, folds, kernels, x);
    
    result = bayesopt(objective_function, [parameter_kernel, parameter_nu, parameter_C]);
    hyperparameters = table2struct(result.XAtMinObjective);
    
    kernel = hyperparameters.kernel;
    nu = hyperparameters.nu;
    C = hyperparameters.C;
end

function mse = crossvalidation_error(inputs, outputs, folds, kernels, hyperparameters)
% Computes the cross-validation MSE for this dataset and hyperparameters
    hyperparameters = table2struct(hyperparameters);
    dataset_partition = kfolds_partition(size(inputs, 1), folds);
    validation_mse = zeros(1, folds);  % keeps the validation MSE for each fold

    for fold_index = 1:folds
        % pick one fold at a time for validation, leave the others for training
        training_inputs = inputs((dataset_partition ~= fold_index),:);
        training_outputs = outputs((dataset_partition ~= fold_index),:);
        validation_inputs = inputs((dataset_partition == fold_index), :);
        validation_outputs = outputs((dataset_partition == fold_index),:);
        % train and estimate the validation MSE for this partition
        model = svr_train(training_inputs, training_outputs, kernels(hyperparameters.kernel), hyperparameters.nu, hyperparameters.C);
        predictions = svr_predict(model, validation_inputs);
        validation_mse(fold_index) = sum((predictions - validation_outputs) .^ 2) / length(validation_outputs);
    end

    mse = mean(validation_mse);
end

function indices = kfolds_partition(N, k)
% Creates the dataset labels for a k-fold partition over N samples
    I = reshape(repmat(1:k, 1, ceil(N/k)), 1, []);
    indices = I(1:N);
end