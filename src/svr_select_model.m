function [kernel, nu, C] = svr_select_model(inputs, outputs, folds, kernels)
% SVR_SELECT_MODEL Selects the hyperparameters of a n-SVR via cross-validation
%
% SYNOPSIS: [kernel, nu, C] = svr_select_model(inputs, outputs, folds, kernels)
%
% INPUT:
% - inputs: a matrix containing one input sample per row
% - outputs: a column vector containing one output sample per entry
% - folds: the number of folds for the cross-validation
% - kernels: a cell array of kernel functions
%
% OUTPUT:
% - kernel: the best kernel function selected
% - nu: the best value selected for this hyperparameter
% - C: the best value selected for this hyperparameter
%
% REMARKS This implementation uses a grid search to find the optimal settings
%
% SEE ALSO svm_select_model, svr_train, svr_predict

    best_mse = +Inf;
    dataset_partition = kfolds_partition(size(inputs, 1), folds);

    for kernel_index = 1:length(kernels)
        % try different kernel functions

        % graph of the MSE function
        figure(kernel_index);
        hold on;
        grid on;
        xlabel('\nu');
        ylabel('C');
        zlabel('MSE');
        set(gca,'YScale','log');

        for try_nu = 0:0.05:1
            for try_C = logspace(-2,6)
                % try different values of the hyperparameters
                
                validation_mse = zeros(1, folds);  % keeps the validation MSE for each fold
                
                for fold_index = 1:folds
                    % pick one fold at a time for validation, leave the others for training
                    training_inputs = inputs((dataset_partition ~= fold_index),:);
                    training_outputs = outputs((dataset_partition ~= fold_index),:);
                    validation_inputs = inputs((dataset_partition == fold_index), :);
                    validation_outputs = outputs((dataset_partition == fold_index),:);
                    % train and estimate the validation MSE for this partition
                    model = svr_train(training_inputs, training_outputs, kernels{kernel_index}, try_nu, try_C);
                    predictions = svr_predict(model, validation_inputs);
                    validation_mse(fold_index) = sum((predictions - validation_outputs) .^ 2) / length(validation_outputs);
                end
                
                mean_validation_mse = mean(validation_mse);
                plot3(try_nu, try_C, mean_validation_mse, 'b.');
                
                if mean_validation_mse < best_mse
                    % we've found a new better model from these hyperparameter settings
                    best_mse = mean_validation_mse;
                    kernel = kernels{kernel_index};
                    nu = try_nu;
                    C = try_C;
                end
            end
        end
    end
end

function indices = kfolds_partition(N, k)
% Creates the dataset labels for a k-fold partition over N samples
    I = reshape(repmat(1:k, 1, ceil(N/k)), 1, []);
    indices = I(1:N);
end