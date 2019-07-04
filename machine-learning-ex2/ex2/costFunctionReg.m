function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% ====================== cost =======================
for i = 1:size(X, 1)
    hx = sigmoid(X(i, :) * theta);
    J += (log(hx) * (-y(i)) - (1 - y(i)) * (log(1 - hx))) / m;
end

for j = 2:size(theta, 1)
    J += (lambda / 2 / m) * (theta(j)^2);
end

% ====================== gradient =======================
for j = 1:size(grad, 1) %same as size(X, 2)
    tmp = 0;
    for i = 1:size(X, 1)
        tmp += (sigmoid(X(i, :) * theta) - y(i)) * X(i, j) / m;
    end
    
    if j == 1
        grad(j) = tmp;
    else 
        grad(j) = tmp + lambda / m * theta(j);
    endif
end

% =============================================================

end
