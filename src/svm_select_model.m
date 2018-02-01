function [kernel, C, best_train_accuracy, Cs] = svm_select_model(train_features, train_classes, test_features, test_classes, folds, kernels, C_range, algorithm, varargin)
% SVM_SELECT_MODEL Selects the hyperparameters of a SVM via cross-validation
%
% SYNOPSIS: [kernel, C, best_accuracy] = svm_select_model(features, classes, folds, kernels, C_range, algorithm)
%           [kernel, C, best_accuracy] = svm_select_model(features, classes, folds, kernels, C_range, 'bundleizator', precision)
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
% - best_accuracy: exactly what you expect it is
%
% REMARKS This implementation uses a grid search to find the optimal settings
%
% SEE ALSO svr_select_model, svm_train, svm_predict

dir =  sprintf('./testdata/svm_model_select/%s',char(datetime()));
mkdir(dir);

best_train_accuracy = -Inf;
best_train_accuracies = containers.Map(kernels.keys, repmat(-Inf, 1, kernels.Count));
Cs = containers.Map(kernels.keys, zeros(1, kernels.Count)); 

dataset_partition = kfolds_partition(size(train_features, 1), folds);

for kernel_index = kernels.keys
    % try different kernel functions

    % graph of the accuracy function
    figure('Name' , kernel_index{1});
    ax1 = subplot(2,1,1);
    %hold on;
    grid on;
    xlabel('C');
    ylabel('accuracy');
    title(kernel_index{1});
    set(ax1, 'XScale', 'log');
    ax2 = subplot(2,1,2);
    xlabel('C');
    ylabel('accuracy');
    
    filename = sprintf('%s/%s.mat', dir, kernel_index{1});
    save(filename, 'algorithm', 'C_range');
    if ~isempty(varargin)
          v = varargin{1};
          save(filename, 'v', '-append');
    end
    
    
    mean_validation_accuracy = zeros(length(C_range),1);
    devi_validation_accuracy = zeros(length(C_range),1);
    test_accuracy = zeros(length(C_range),1);
    test_accuracy_pos = zeros(length(C_range),1);
    test_accuracy_neg = zeros(length(C_range),1);
    
    for i = 1:length(C_range)
        % try different values of the hyperparameter C
        try_C = C_range(i);        
        
        % keeps the validation accuracy for each fold
        validation_accuracy = zeros(1, folds);

        for fold_index = 1:folds
            % pick one fold at a time for validation, leave the others for training
            training_features = train_features((dataset_partition ~= fold_index),:);
            training_classes = train_classes((dataset_partition ~= fold_index),:);
            validation_features = train_features((dataset_partition == fold_index), :);
            validation_classes = train_classes((dataset_partition == fold_index),:);

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
        
        mean_validation_accuracy(i) = mean(validation_accuracy);
        devi_validation_accuracy(i) = std(validation_accuracy,1);
        
        %Compute the accuracy on the given test data (just for plots)
        prediction = svm_predict(model, test_features, algorithm);
        test_accuracy(i) = sum(prediction == test_classes)/length(test_classes);
        q0 = prediction == test_classes & test_classes == -1;
        q1 = prediction == test_classes & test_classes == 1;
        test_accuracy_neg(i) = sum(q0)/length(test_classes);
        test_accuracy_pos(i) = sum(q1)/length(test_classes);
        
        save(filename, 'mean_validation_accuracy', 'devi_validation_accuracy', 'test_accuracy', 'test_accuracy_pos', 'test_accuracy_neg', '-append');
        
        
        errorbar(ax1, C_range, mean_validation_accuracy, devi_validation_accuracy, 'c', 'CapSize', 0);
        set(ax1,'NextPlot','add');
        plot(ax1, C_range, mean_validation_accuracy, '-b', C_range, test_accuracy, ':r');
        ax1.YLabel.String = 'accuracy';
        set(ax1, 'XLim', [C_range(1) C_range(end)]);
        set(ax1, 'YLim', [0.4 1]);
        set(ax1, 'XScale', 'log');
        set(ax1, 'YTick', 0.4:.2:1);
        ax1.YGrid = 'on';
        ax1.XGrid = 'on';
        set(ax1, 'NextPlot', 'replace');
        
        bar(ax2,[test_accuracy_pos test_accuracy_neg], 'stacked', 'EdgeColor', 'flat');
        ax2.XLabel.String = 'C';
        ax2.YLabel.String = 'accuracy';
        set(ax2, 'XTick', 1:10:100);
        set(ax2, 'XTickLabel', {''});
        set(ax2, 'YTick', 0:.2:1);
        set(ax2, 'XLim', [1 length(C_range)]);
        set(ax2, 'YLim', [0 1]);
        
        drawnow;

        if mean_validation_accuracy(i) > best_train_accuracies(kernel_index{1})
            % we've found better hyperparameter settings for this kernel
            best_train_accuracies(kernel_index{1}) = mean_validation_accuracy(i);
            Cs(kernel_index{1}) = try_C;
            save(filename, 'Cs', 'best_train_accuracies', '-append');
        end
    end
          
    fprintf('Kernel: %s, C: %e, train accuracy: %f\n', kernel_index{1}, Cs(kernel_index{1}), best_train_accuracies(kernel_index{1}));
    set(ax1,'NextPlot','add');
    plot(ax1, Cs(kernel_index{1}), best_train_accuracies(kernel_index{1}), '*', 'LineWidth', 1); 
    set(ax1,'NextPlot','replace');
    
    if best_train_accuracies(kernel_index{1}) > best_train_accuracy
        % we've found a better performance on this kernel
        best_train_accuracy = best_train_accuracies(kernel_index{1});
        kernel = kernel_index{1};
        C = Cs(kernel_index{1});
        save(sprintf('%s/best.mat', dir), 'best_train_accuracy', 'kernel', 'C');
    end
    
   
    
end

end