% data set

n = 40;
omega = randn(1, 1);

noise = 0.8 * randn(n, 1);
x = randn(n, 2);
y = 2 * (omega * x(:, 1) + x(:, 2) + noise > 0) - 1;

hold off
scatter(x(:, 1), x(:, 2), 16, y);

% initial value

alpha = randn(n, 1);
alpha(alpha > 1) = 1;
alpha(alpha < 0) = 0;

K = zeros(n, n);
ita = 0.1;
lambda = 1;

% x(i, :) is a row vector, so its transpose is Xi
% so it is x(i, :) * x(j, :)' for Xi'Xj.

for i = 1 : 40
    for j = 1 : 40
        K(i, j) = y(i) * y(j) * (x(i, :) * x(j, :)');
    end
end

dual = zeros(100, 1);
loss = zeros(100, 1);

for iteration = 1 : 100
    alpha = alpha - ita * ((1 / (2 * lambda)) * K * alpha - 1);
    
    alpha(alpha > 1) = 1;
    alpha(alpha < 0) = 0;
    
    dual(iteration) = - ((1 / (4 * lambda)) * alpha' * K * alpha) + alpha' * ones(n, 1);
    
    opti_w = zeros(2, 1);
    
    for i = 1 : n
        opti_w = opti_w + ((1 / (2 * lambda)) * (alpha(i) * y(i) * x(i, :)'));
    end
    
    loss(iteration) = 0;
    for i = 1 : n
        loss(iteration) = loss(iteration) + max(0, 1 - (y(i) * opti_w' * x(i, :)'));
    end
    loss(iteration) = loss(iteration) + lambda * (norm(opti_w) ^ 2);
    
    disp("*****")
    disp("Iteration " + iteration);
    disp("Dual = " + dual(iteration));
    disp("Loss = " + loss(iteration));
end

hold off
plot(dual)
hold on
plot(loss)
