# Re-importing numpy as the execution state was reset
import numpy as np

# Example data
X = np.array([1, 2, 3, 4, 5])  # Feature
Y = np.array([2, 3, 4, 5, 6])  # Target

# Initial parameters of the model
theta_0 = 0.0  # Intercept
theta_1 = 1.0  # Slope

# Learning rate
alpha = 0.01

# Number of training examples
m = len(X)

# Cost function (Mean Squared Error)
def compute_cost(X, Y, theta_0, theta_1):
    predictions = theta_0 + theta_1 * X
    errors = predictions - Y
    squared_errors = errors ** 2
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    return cost

# Initial cost
initial_cost = compute_cost(X, Y, theta_0, theta_1)

# Gradient Descent Function to update theta_0 and theta_1
def gradient_descent(X, Y, theta_0, theta_1, alpha, iterations):
    cost_history = [0] * iterations
    
    for iteration in range(iterations):
        predictions = theta_0 + theta_1 * X
        
        error_theta_0 = predictions - Y
        error_theta_1 = (predictions - Y) * X
        
        theta_0 -= alpha * (1.0 / m) * np.sum(error_theta_0)
        theta_1 -= alpha * (1.0 / m) * np.sum(error_theta_1)
        
        cost = compute_cost(X, Y, theta_0, theta_1)
        cost_history[iteration] = cost
        
    return theta_0, theta_1, cost_history

# Perform 1 iteration of gradient descent
theta_0, theta_1, cost_history = gradient_descent(X, Y, theta_0, theta_1, alpha, 1)

# Updated parameters and cost
updated_theta_0, updated_theta_1, current_cost = theta_0, theta_1, cost_history[-1]

initial_cost, updated_theta_0, updated_theta_1, current_cost
