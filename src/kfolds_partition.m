function indices = kfolds_partition(N, k)
% Create dataset labels for a k-fold partition over N samples
    I = reshape(repmat(1:k, 1, ceil(N/k)), 1, []);
    indices = I(1:N);
end
