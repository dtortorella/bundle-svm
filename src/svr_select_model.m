function [kernel, C, epsilon, mean_validation_mse, devi_validation_mse] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range)
% SVR_SELECT_MODEL Selects the hyperparameters of a SVR via cross-validation
%
% SYNOPSIS: [kernel, C, epsilon] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a column vector containing one output sample per entry
% - folds: the number of folds for the cross-validation
% - kernels: a cell array of kernel functions
% - C_range: a vector containing C values to try
% - epsilon_range: a vector containing epsilon values to try
%
% OUTPUT:
% - kernel: the best kernel function selected
% - C: the best value selected for this hyperparameter
% - epsilon: the best value selected for this hyperparameter
%
% REMARKS This implementation uses a grid search to find the optimal settings
%
% SEE ALSO svm_select_model, svr_train, svr_predict

    best_mse = +Inf;
    dataset_partition = kfolds_partition(size(inputs, 1), folds);

    for kernel_index = 1:length(kernels)
        % try different kernel functions

        % graph of the MSE function
        figure%(kernel_index);
        %hold on;
        grid on;
        xlabel('C');
        ylabel('\epsilon');
        zlabel('MSE');
        set(gca,'XScale','log');

        mean_validation_mse = zeros(length(C_range), length(epsilon_range));
        devi_validation_mse= zeros(length(C_range), length(epsilon_range));
        
        for i = 1:length(C_range)
            fprintf('\nC = %f, eps = ', C_range(i));
            for j = 1:length(epsilon_range)
                % try different values of the hyperparameters
                fprintf('%f ', epsilon_range(j));
                validation_mse = zeros(1, folds);  % keeps the validation MSE for each fold
                
                failure = false;
                
                for fold_index = 1:folds
                    % pick one fold at a time for validation, leave the others for training
                    training_inputs = inputs((dataset_partition ~= fold_index),:);
                    training_outputs = outputs((dataset_partition ~= fold_index),:);
                    validation_inputs = inputs((dataset_partition == fold_index), :);
                    validation_outputs = outputs((dataset_partition == fold_index),:);
                    % train and estimate the validation MSE for this partition
                    model = svr_train(training_inputs, training_outputs, kernels{kernel_index}, C_range(i), epsilon_range(j));
                    
                   failure = ~isstruct(model);
                   if failure 
                       break
                   end
                    predictions = svr_predict(model, validation_inputs);
                    validation_mse(fold_index) = sum((predictions - validation_outputs) .^ 2) / length(validation_outputs);
                end
                
               if failure 
                   continue
               end
                mean_validation_mse(i,j) = mean(validation_mse);
                devi_validation_mse(i,j) = std(validation_mse);

                surf(C_range, epsilon_range, mean_validation_mse', devi_validation_mse','FaceColor', 'interp');
                drawnow
 
                if mean_validation_mse(i,j) < best_mse
                    % we've found a new better model from these hyperparameter settings
                    best_mse = mean_validation_mse(i,j);
                    kernel = kernels{kernel_index};
                    C = C_range(i);
                    epsilon = epsilon_range(j);
                end
            end
        end
    end
      
end