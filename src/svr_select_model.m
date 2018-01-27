function [kernel, C, epsilon, best_mee, Cs, epsilons] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range, algorithm, varargin)
% SVR_SELECT_MODEL Selects the hyperparameters of a SVR via cross-validation
%
% SYNOPSIS: [kernel, C, epsilon, best_mee] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range, algorithm)
%           [kernel, C, epsilon, best_mee] = svr_select_model(inputs, outputs, folds, kernels, C_range, epsilon_range, 'bundleizator', precision)
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
% - best_mee: exactly what you expect it is
%
% REMARKS This implementation uses a grid search to find the optimal settings
%
% SEE ALSO svm_select_model, svr_train, svr_predict

best_mee = +Inf;
best_mees = containers.Map(kernels.keys, repmat(+Inf, 1, kernels.Count));
Cs = containers.Map(kernels.keys, zeros(1, kernels.Count));
epsilons = containers.Map(kernels.keys, zeros(1, kernels.Count));
dataset_partition = kfolds_partition(size(inputs, 1), folds);

for kernel_index = kernels.keys
    % try different kernel functions

    % graph of the MEE function
    fig = figure('Name', kernel_index{1});
    grid on;
    xlabel('C');
    ylabel('\epsilon');
    zlabel('MEE');
    title(kernel_index{1});
    set(gca, 'XScale', 'log');

    mean_validation_mee = zeros(length(C_range), length(epsilon_range));
    devi_validation_mee= zeros(length(C_range), length(epsilon_range));

    % try different values of the hyperparameters
    for i = 1:length(C_range)
        for j = 1:length(epsilon_range)
            
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

            figure(fig);
            surf(C_range, epsilon_range, mean_validation_mee', devi_validation_mee', 'FaceColor', 'interp');
            grid on;
            xlabel('C');
            ylabel('\epsilon');
            zlabel('MEE');
            title(kernel_index{1});
            set(gca, 'XScale', 'log');
            drawnow;

            if mean_validation_mee(i,j) < best_mees(kernel_index{1})
                % we've found better hyperparameter settings for this kernel
                best_mees(kernel_index{1}) = mean_validation_mee(i,j);
                kernel = kernels(kernel_index{1});
                Cs(kernel_index{1}) = C_range(i);
                epsilons(kernel_index{1}) = epsilon_range(j);
            end
        end
    end
    
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