function [kernel, nu] = svm_select_model(features, classes, folds, kernels, nu_range)
% SVM_SELECT_MODEL Selects the hyperparameters of a n-SVM via cross-validation
%
% SYNOPSIS: [kernel, nu] = svm_select_model(features, classes, folds, kernels, nu_range)
%
% INPUT:
% - features: a matrix containing one sample feature vector per row
% - classes: a column vector containing one sample class per entry, must be +/-1
% - folds: the number of folds for the cross-validation
% - kernels: a cell array of kernel functions
% - nu_range: a vector containing the nu values to try
%
% OUTPUT:
% - kernel: the best kernel function selected
% - nu: the best value selected for this hyperparameter
%
% REMARKS This implementation uses a grid search to find the optimal settings
%
% SEE ALSO svr_select_model, svm_train, svm_predict

    best_accuracy = -Inf;
    dataset_partition = kfolds_partition(size(features, 1), folds);

    for kernel_index = 1:length(kernels)
        % try different kernel functions
        
        % graph of the accuracy function
        figure%(kernel_index);
        hold on;
        grid on;
        xlabel('\nu');
        ylabel('accuracy');

        for try_nu = nu_range
            % try different values of the hyperparameter nu

            validation_accuracy = zeros(1, folds);  % keeps the validation accuracy for each fold
            failure = false;

            for fold_index = 1:folds
                % pick one fold at a time for validation, leave the others for training
                training_features = features((dataset_partition ~= fold_index),:);
                training_classes = classes((dataset_partition ~= fold_index),:);
                validation_features = features((dataset_partition == fold_index), :);
                validation_classes = classes((dataset_partition == fold_index),:);
                
                % train and estimate the validation accuracy for this partition
                model = svm_train(training_features, training_classes, kernels{kernel_index}, try_nu);
                
                failure = ~isstruct(model); %check if the training found a solution
                if ~failure %do not continue if no model was found
                    predictions = svm_predict(model, validation_features);
                    validation_accuracy(fold_index) = sum(predictions == validation_classes) / length(validation_classes);
                else
                    break
                end
                
            end
            
            if failure
                continue
            end
            
            mean_validation_accuracy = mean(validation_accuracy);
            plot(try_nu, mean_validation_accuracy, 'b.');
            errorbar(try_nu, mean_validation_accuracy, std(validation_accuracy,1), 'c', 'CapSize', 0);

            if mean_validation_accuracy > best_accuracy
                % we've found a new better model from these hyperparameter settings
                best_accuracy = mean_validation_accuracy;
                kernel = kernels{kernel_index};
                nu = try_nu;
            end
        end
    end
end

function indices = kfolds_partition(N, k)
% Create dataset labels for a k-fold partition over N samples
    I = reshape(repmat(1:k, 1, ceil(N/k)), 1, []);
    indices = I(1:N);
end