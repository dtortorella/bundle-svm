function test_bundleizator_monk(X, y, X_test, y_test, C, kernel, bundle_precision)
%TEST_BUNDLE_MONK Summary of this function goes here
%   Detailed explanation goes here

acc = zeros(size(C));
acc_test = zeros(size(C));
iters = zeros(size(C));

figure;
title('train error given C')
set(gca,'XScale','log');
%hold on;
grid on;

for k = 1:length(C)
    
    [u, iters(k)] = bundleizator(X, y, C(k), kernel, @hinge_loss, @hinge_dloss, bundle_precision);
    
    %% Training accuracy
    
    yBundle = zeros(size(y));
    for i = 1:size(X,1)
        yBundle(i) = bundleizator_classify(X(i,:), X, kernel, u);
    end
        
    q = yBundle == y;
    acc(k) = sum(q)/length(yBundle);
        
    %% Test accuracy
    
    yBundle = zeros(size(y_test));
    for i = 1:size(X_test,1)
        yBundle(i) = bundleizator_classify(X_test(i,:), X, kernel, u);
    end
        
    q = yBundle == y_test;
    acc_test(k) = sum(q)/length(yBundle);
    
    %% Plot
    
    subplot(2,1,1);
    plot(C, acc, 'b.', C ,acc_test, 'r.');
    set(gca,'XScale','log');
    grid on;
    
    subplot(2,1,2);
    bar(iters);
    grid on;
    
    drawnow;
end

end