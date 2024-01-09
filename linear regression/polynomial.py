import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Fit a polynomial of degree 2 (quadratic) to the data
p = Polynomial.fit(x, y, 2)

# Get the coefficients of the polynomial
coefs = p.convert().coef

# Print the polynomial equation
print(f'The polynomial regression equation is: y = {coefs[2]}x^2 + {coefs[1]}x + {coefs[0]}')

# Function to predict y using the polynomial model
def predict(x):
    return coefs[2] * x**2 + coefs[1] * x + coefs[0]

# Generate a range of values for x
x_new = np.linspace(min(x), max(x), 100)
# Predict y for the new x values
y_new = predict(x_new)

# Plot the original data and the polynomial curve
plt.scatter(x, y, color='red', label='Data points')
plt.plot(x_new, y_new, label='Polynomial fit')
plt.legend()
plt.show()
