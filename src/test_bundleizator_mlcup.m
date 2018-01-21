function test_bundleizator_mlcup(X, y, X_test, y_test, C, epsilon, kernel, bundle_precision)
%TEST_BUNDLEIZATOR_MLCUP Summary of this function goes here
%   Detailed explanation goes here

acc = zeros(length(epsilon),length(C));
acc_test = zeros(length(epsilon), length(C));
iters = zeros(length(epsilon), length(C));

figure;
title('train error given C')
set(gca,'XScale','log');
%hold on;
grid on;

for j = 1:length(epsilon)
    for k = 1:length(C)

        [u, iters(j,k)] = bundleizator(X, y, C(k), kernel, @(f,y) einsensitive_loss(f, y, epsilon(j)), @(f,y) einsensitive_dloss(f, y, epsilon(j)), bundle_precision, 1e-12);

        %% Training accuracy

        yBundle = zeros(size(y));
        for i = 1:size(X,1)
            yBundle(i) = bundleizator_predict(X(i,:), X, kernel, u);
        end

        acc(j,k) = sum((yBundle - y) .^ 2) / length(y);

        %% Test accuracy

        yBundle = zeros(size(y_test));
        for i = 1:size(X_test,1)
            yBundle(i) = bundleizator_predict(X_test(i,:), X, kernel, u);
        end

        acc_test(j,k) = sum((yBundle - y_test) .^ 2) / length(y_test);

        %% Plot

%        subplot(2,1,1);
        surf(C, epsilon, acc,'FaceColor', 'interp');
        %plot3(C, epsilon, acc, 'b.', 'r.');
        %set(gca,'YScale','log');
        grid on;

%         subplot(2,1,2);
%         surf(C, epsilon, acc_test,'FaceColor', 'interp');
%         grid on;
%         set(gca,'YScale','log');
        drawnow;
       
    end
end

end


