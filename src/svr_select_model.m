function [kernel, C, epsilon, mean_validation_mee, devi_validation_mee] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range, algorithm, varargin)
% SVR_SELECT_MODEL Selects the hyperparameters of a SVR via cross-validation
%
% SYNOPSIS: [kernel, C, epsilon] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range, algorithm)
%           [kernel, C, epsilon] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range, 'bundleizator', precision)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a matrix containing one output sample vector per row
% - folds: the number of folds for the cross-validation
% - kernels: a map of strings -> kernel functions
% - C_range: a vector containing C values to try
% - epsilon_range: a vector containing epsilon values to try
% - algorithm: which implementation to use (bundleizator/libsvm)
% - precision: the required distance from optimality (optional, only for bundleizator)
%
% OUTPUT:
% - kernel: the best kernel function selected
% - C: the best value selected for this hyperparameter
% - epsilon: the best value selected for this hyperparameter
%
% REMARKS This implementation uses a grid search to find the optimal settings
%
% SEE ALSO svm_select_model, svr_train, svr_predict

best_mee = +Inf;
dataset_partition = kfolds_partition(size(inputs, 1), folds);

for kernel_index = kernels.keys
    % try different kernel functions

    % graph of the MEE function
    figure;
    grid on;
    xlabel('C');
    ylabel('\epsilon');
    zlabel('MEE');
    set(gca, 'XScale', 'log');

    mean_validation_mee = zeros(length(C_range), length(epsilon_range));
    devi_validation_mee= zeros(length(C_range), length(epsilon_range));

    % try different values of the hyperparameters
    for i = 1:length(C_range)
        fprintf('\nC = %f, eps = ', C_range(i));
        for j = 1:length(epsilon_range)
            fprintf('%f ', epsilon_range(j));
            
            % keeps the validation MEE for each fold
            validation_mee = zeros(1, folds);

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
                        model = svr_train(training_inputs, training_outputs(:,k), kernels(kernel_index{1}), C_range(i), epsilon_range(j), algorithm);
                    else
                        model = svr_train(training_inputs, training_outputs(:,k), kernels(kernel_index{1}), C_range(i), epsilon_range(j), algorithm, varargin{1});
                    end
                    predictions(:,k) = svr_predict(model, validation_inputs, algorithm);
                end
                
                % estimate the validation MEE for this partition
                validation_mee(fold_index) = sum(sqrt(sum(((predictions - validation_outputs) .^ 2), 2))) / length(validation_outputs);
            end
            
            mean_validation_mee(i,j) = mean(validation_mee);
            devi_validation_mee(i,j) = std(validation_mee, 1);

            surf(C_range, epsilon_range, mean_validation_mee', devi_validation_mee', 'FaceColor', 'interp');
            set(gca, 'XScale', 'log');
            drawnow;

            if mean_validation_mee(i,j) < best_mee
                % we've found a new better model from these hyperparameter settings
                best_mee = mean_validation_mee(i,j);
                kernel = kernels(kernel_index{1});
                C = C_range(i);
                epsilon = epsilon_range(j);
            end
        end
    end
end
      
end