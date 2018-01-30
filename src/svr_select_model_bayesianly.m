function [kernel, C, epsilon, best_mee, Cs, epsilons] = svr_select_model_bayesianly(inputs, outputs, folds, kernels, C_range, epsilon_range, algorithm, varargin)
% SVR_SELECT_MODEL_BAYESIANLY Selects the hyperparameters of a SVR via cross-validation
%
% SYNOPSIS: [kernel, C, epsilon] = svr_select_model_bayesianly(inputs, outputs, folds, kernels)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a matrix containing one output sample vector per row
% - folds: the number of folds for the cross-validation
% - kernels: a map of strings -> kernel functions
%
% OUTPUT:
% - kernel: the best kernel function selected
% - C: the best value selected for this hyperparameter
% - epsilon: the best value selected for this hyperparameter
%
% REMARKS This implementation uses bayesian optimization to find the best settings
%
% SEE ALSO svr_select_model, svm_select_model_bayesianly

function mee = crossvalidation_error(kernel_func, hyperparameters)
    % Computes the cross-validation MEE for this dataset and hyperparameters
    hyperparameters = table2struct(hyperparameters);
    dataset_partition = kfolds_partition(size(inputs, 1), folds);
    validation_mee = zeros(1, folds);  % keeps the validation MEE for each fold

    for fold_index = 1:folds
        % pick one fold at a time for validation, leave the others for training
        training_inputs = inputs((dataset_partition ~= fold_index),:);
        training_outputs = outputs((dataset_partition ~= fold_index),:);
        validation_inputs = inputs((dataset_partition == fold_index),:);
        validation_outputs = outputs((dataset_partition == fold_index),:);

        % train the regression model for each output component and predict outputs
        predictions = zeros(size(validation_outputs));
        for k = 1:size(outputs, 2)
            if isempty(varargin)
                model = svr_train(training_inputs, training_outputs, kernel_func, hyperparameters.C, hyperparameters.epsilon, algorithm);
            else
                model = svr_train(training_inputs, training_outputs, kernel_func, hyperparameters.C, hyperparameters.epsilon, algorithm, varargin{1});
            end
            predictions(:,k) = svr_predict(model, validation_inputs, algorithm);
        end

        % estimate the validation MEE for this partition
        validation_mee(fold_index) = sum(sqrt(sum(((predictions - validation_outputs) .^ 2), 2))) / length(validation_outputs);
    end

    mee = mean(validation_mee);
end
    
best_mee = Inf;
best_mees = containers.Map(kernels.keys, repmat(Inf, 1, kernels.Count));
Cs = containers.Map(kernels.keys, zeros(1, kernels.Count));
epsilons = containers.Map(kernels.keys, zeros(1, kernels.Count));

parameter_C = optimizableVariable('C', [C_range(1), C_range(end)], 'Type', 'real', 'Transform', 'log');
parameter_epsilon = optimizableVariable('epsilon', [epsilon_range(1), epsilon_range(end)], 'Type', 'real');

for kernel_index = kernels.keys
    % optimize for the current kernel
    objective_function = @(x) crossvalidation_error(kernels(kernel_index{1}), x);
    result = bayesopt(objective_function, [parameter_C, parameter_epsilon], 'MaxObjectiveEvaluations', 20);
    
    % store the current optimum
    best_mees(kernel_index{1}) = result.MinObjective;
    best_hyperparameters = table2struct(result.XAtMinObjective);
    Cs(kernel_index{1}) = best_hyperparameters.C;
    epsilons(kernel_index{1}) = best_hyperparameters.epsilon;

    fprintf('Kernel: %s, C: %e, epsilon: %f, MEE: %f\n', kernel_index{1}, Cs(kernel_index{1}), epsilons(kernel_index{1}), best_mees(kernel_index{1}));

    if best_mees(kernel_index{1}) < best_mee
        % we've found a better performance on this kernel
        best_mee = best_mees(kernel_index{1});
        kernel = kernel_index{1};
        C = Cs(kernel_index{1});
        epsilon = epsilons(kernel_index{1});
    end
end

end