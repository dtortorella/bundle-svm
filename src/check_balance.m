function check_balance(classes, folds)
    partition = balanced_kfolds_partition(classes, folds);
    for i = 1:folds
        fprintf('%d \\ %d \t', sum(classes((partition == i))), length(classes((partition == i))));
    end
    fprintf('\n');
end