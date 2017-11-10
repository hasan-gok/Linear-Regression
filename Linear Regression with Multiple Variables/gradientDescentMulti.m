function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_temp = theta;
theta_size = size(theta);
index = 1;
for iter = 1:num_iters
    theta_temp(index) = theta(index) - alpha * ((X * theta - y)'*X(:,index))/m;
    index = index + 1;
    if index > theta_size
        index = 1;
    end
    theta = theta_temp;
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
