function [kernel, C] = svm_select_model(features, classes, folds, kernels, C_range, algorithm, varargin)
% SVM_SELECT_MODEL Selects the hyperparameters of a SVM via cross-validation
%
% SYNOPSIS: [kernel, C] = svm_select_model(features, classes, folds, kernels, C_range)
%
% INPUT:
% - features: a matrix containing one sample feature vector per row
% - classes: a column vector containing one sample class per entry, must be +/-1
% - folds: the number of folds for the cross-validation
% - kernels: a map of strings -> kernel functions
% - C_range: a vector containing the C values to try
% - algorithm: which implementation to use (bundleizator/libsvm)
% - precision: the required distance from optimality (optional, only for bundleizator)
%
% OUTPUT:
% - kernel: the best kernel function selected
% - C: the best value selected for this hyperparameter
%
% REMARKS This implementation uses a grid search to find the optimal settings
%
% SEE ALSO svr_select_model, svm_train, svm_predict

best_accuracy = -Inf;
dataset_partition = kfolds_partition(size(features, 1), folds);

for kernel_index = kernels.keys
    % try different kernel functions

    % graph of the accuracy function
    figure;
    hold on;
    grid on;
    xlabel('C');
    ylabel('accuracy');
    title(kernel_index{1});
    set(gca, 'XScale', 'log');

    for try_C = C_range
        % try different values of the hyperparameter C

        % keeps the validation accuracy for each fold
        validation_accuracy = zeros(1, folds);

        for fold_index = 1:folds
            % pick one fold at a time for validation, leave the others for training
            training_features = features((dataset_partition ~= fold_index),:);
            training_classes = classes((dataset_partition ~= fold_index),:);
            validation_features = features((dataset_partition == fold_index), :);
            validation_classes = classes((dataset_partition == fold_index),:);

            % train the model for the current dataset partion 
            if isempty(varargin)
                model = svm_train(training_features, training_classes, kernels(kernel_index{1}), try_C, algorithm);
            else
                model = svm_train(training_features, training_classes, kernels(kernel_index{1}), try_C, algorithm, varargin{1});
            end

            % estimate the validation accuracy for this partition
            predictions = svm_predict(model, validation_features, algorithm);
            validation_accuracy(fold_index) = sum(predictions == validation_classes) / length(validation_classes);
        end

        mean_validation_accuracy = mean(validation_accuracy);
        errorbar(try_C, mean_validation_accuracy', std(validation_accuracy,1), 'c', 'CapSize', 0);
        plot(try_C, mean_validation_accuracy, 'b.');
        drawnow;

        if mean_validation_accuracy > best_accuracy
            % we've found a new better model from these hyperparameter settings
            best_accuracy = mean_validation_accuracy;
            kernel = kernel_index{1};
            C = try_C;
        end
    end
end

end