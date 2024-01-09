import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Calculate the means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the slope b
b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)

# Calculate the intercept a
a = y_mean - b * x_mean

# Display the linear regression equation
equation = f'y = {a} + {b}x'
print("Linear regression equation:", equation)

# Function to predict y for a given x using the linear regression equation
def predict(x):
    return a + b * x

# Predict y for a new x value
x_new = 6
y_predicted = predict(x_new)
print(f"The predicted value of y for x = {x_new} is: {y_predicted}")
