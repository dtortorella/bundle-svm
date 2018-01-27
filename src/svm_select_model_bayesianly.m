function [kernel, C, best_accuracy, Cs, best_accuracies] = svm_select_model_bayesianly(features, classes, folds, kernels, C_range, algorithm, varargin)
% SVM_SELECT_MODEL_BAYESIANLY Selects the hyperparameters of a SVM via cross-validation
%
% SYNOPSIS: [kernel, C] = svm_select_model_bayesianly(features, classes, folds, kernels, algorithm)
%           [kernel, C] = svm_select_model_bayesianly(features, classes, folds, kernels, 'bundleizator', precision)
%
% INPUT:
% - features: a matrix containing one sample feature vector per row
% - classes: a column vector containing one sample class per entry, must be +/-1
% - folds: the number of folds for the cross-validation
% - kernels: a map of strings -> kernel functions
% - algorithm: which implementation to use (bundleizator/libsvm)
% - precision: the required distance from optimality (optional, only for bundleizator)
%
% OUTPUT:
% - kernel: the best kernel function selected
% - C: the best value selected for this hyperparameter
%
% REMARKS This implementation uses bayesian optimization to find the best settings
%
% SEE ALSO svm_select_model, svr_select_model_bayesianly

function inaccuracy = crossvalidation_error(kernel_func, hyperparameters)
    % Computes the cross-validation error (1 - accuracy) for this dataset and hyperparameters
    hyperparameters = table2struct(hyperparameters);
    dataset_partition = kfolds_partition(size(features, 1), folds);
    validation_error = zeros(1, folds);  % keeps the validation inaccuracy for each fold

    for fold_index = 1:folds
        % pick one fold at a time for validation, leave the others for training
        training_features = features((dataset_partition ~= fold_index),:);
        training_classes = classes((dataset_partition ~= fold_index),:);
        validation_features = features((dataset_partition == fold_index),:);
        validation_classes = classes((dataset_partition == fold_index),:);

        % train the model for the current dataset partion 
        if isempty(varargin)
            model = svm_train(training_features, training_classes, kernel_func, hyperparameters.C, algorithm);
        else
            model = svm_train(training_features, training_classes, kernel_func, hyperparameters.C, algorithm, varargin{1});
        end

        % estimate the validation accuracy for this partition
        predictions = svm_predict(model, validation_features, algorithm);
        validation_error(fold_index) = sum(predictions ~= validation_classes) / length(validation_classes);
    end
    
    inaccuracy = mean(validation_error);
end

best_accuracy = -Inf;
best_accuracies = containers.Map(kernels.keys, repmat(-Inf, 1, kernels.Count));
Cs = containers.Map(kernels.keys, zeros(1, kernels.Count));

parameter_C = optimizableVariable('C', [C_range(1), C_range(end)], 'Type', 'real', 'Transform', 'log');

for kernel_index = kernels.keys
    % optimize for the current kernel
    objective_function = @(x) crossvalidation_error(kernels(kernel_index{1}), x);
    result = bayesopt(objective_function, [parameter_C], 'MaxObjectiveEvaluations', 20);

    % store the current optimum
    best_accuracies(kernel_index{1}) = 1 - result.MinObjective;
    best_hyperparameters = table2struct(result.XAtMinObjective);
    Cs(kernel_index{1}) = best_hyperparameters.C;

    fprintf('Kernel: %s, C: %e, accuracy: %f\n', kernel_index{1}, Cs(kernel_index{1}), best_accuracies(kernel_index{1}));

    if best_accuracies(kernel_index{1}) > best_accuracy
        % we've found a better performance on this kernel
        best_accuracy = best_accuracies(kernel_index{1});
        kernel = kernel_index{1};
        C = Cs(kernel_index{1});
    end
end

end