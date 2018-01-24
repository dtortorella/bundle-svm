function [model, TRaccurancy, TSaccurancy] = test_libsvm_monk(Xtrain, ytrain, Xtest, ytest, C)

for k = 1:length(C)

    opt = sprintf('-s 0 -t 1 -d 2 -c %f -q', C(k));

    model(k) = libsvmtrain(ytrain, Xtrain, opt);

    predictions = libsvmpredict(ytrain, Xtrain, model(k), '-q');
    TRaccurancy(k) = sum(predictions == ytrain)/length(predictions);

    predictions= libsvmpredict(ytest, Xtest, model(k), '-q');
    TSaccurancy(k) = sum(predictions == ytest)/length(predictions);
    fprintf("k = %d , C = %e, TRacc = %f, TSacc = %f\n", k, C(k), TRaccurancy(k), TSaccurancy(k));

end

end

