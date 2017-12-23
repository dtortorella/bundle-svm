function nu_star = svm_max_feasible_nu(classes,varargin)
%MAX_FEASIBLE_NU Returns the maximum feasible nu for an nu-SVM classifier

    if isempty(varargin)
        positive_labels = sum(classes > 0);
        negative_labels = sum(classes <= 0);
        nu_star = 2 * min( [positive_labels negative_labels] ) / length(classes);
    elseif length(varargin) == 1
        folds = varargin{1};
        partition = balanced_kfolds_partition(classes, folds);
        for i = 1:folds
            positive_labels = sum(classes(partition == i) > 0);
            negative_labels = sum(classes(partition == i) <= 0);
            nu_star(i) = 2 * min( [positive_labels negative_labels] ) / length(classes(partition == i));
        end
    end
end

