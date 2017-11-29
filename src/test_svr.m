M = importdata("../data/ml-cup17.train.csv");
inputs = M(:,2:end-2);
outputs1 = M(:,end-1);
outputs2 = M(:,end);
proportion = floor(2*length(outputs1)/3);
TR_inputs = inputs(1:proportion,:);
TR_outputs1 = outputs1(1:proportion);
TR_outputs2 = outputs2(1:proportion);
TS_inputs = inputs(proportion+1:end,:);
TS_outputs1 = outputs1(proportion+1:end);
TS_outputs2 = outputs2(proportion+1:end);
% C value computed as the standard value in BoxConstraint of fitrsvm with rbf kernel
[mdl1,epsilon] = svr_train(TR_inputs,TR_outputs1, 2, 0.50, iqr(outputs1)/1.349); %using rbf kernel (see svm_train file)
predictions1 = svr_predict(mdl1, TS_inputs);

