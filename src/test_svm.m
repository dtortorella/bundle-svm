M = importdata("../data/monks-1.train.csv");
c = M(:,1);
classes = c*2 - ones(length(c),1);
features = M(:,2:end);
proportion = floor(2*length(c)/3);
TR_features = features(1:proportion,:);
TR_classes = classes(1:proportion);
TS_features = features(proportion+1:end,:);
TS_classes = classes(proportion+1:end);
mdl = svm_train(TR_features, TR_classes, 2 ,0.50); %using rbf kernel (see svm_train file)
prediction = svm_predict(mdl,TS_features);

