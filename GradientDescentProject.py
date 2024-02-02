# NAME = David Gourley, davgourl@iu.edu | E201

# Twelve data points, each with an x-coordinate and a y-coordinate, are posted to Canvas Files.
# Write original code from scratch that implements the gradient descent method to obtain the line of
# best fit for this data.
# Plot the data points with your line overlayed on them.
# I suggest verifying your result against a pre-written library function.
# I would try a step size of .001

# Data Points:
# A = (2.9, 4.0), (-1.5, -0.9), (0.1, 0.0), (-1.0, -1.0), (2.1, 3.0), (-4.0, -5.0), (-2.0, -3.5), (2.2, 2.6), (0.2, 1.0), (2.0, 3.5), (1.5, 1.0), (-2.5, -4.7)
# X-coordinate = Height
# Y-coordinate = Weight

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_points = np.array([   # data points
    [2.9, 4.0],
    [-1.5, -0.9],
    [0.1, 0.0],
    [-1.0, -1.0],
    [2.1, 3.0],
    [-4.0, -5.0],
    [-2.0, -3.5],
    [2.2, 2.6],
    [0.2, 1.0],
    [2.0, 3.5],
    [1.5, 1.0],
    [-2.5, -4.7]
])

#find slope and y int of line of best fit using gradient_descent function
def gradient_descent(data, learn_rate=0.001, num_iterations=1000):
    m,b = 0,0 

    for _ in range(num_iterations):
        derivative_m = (-2/len(data)) * np.sum(data[:, 0] * (data[:, 1] - (m * data[:, 0] + b)))
        derivative_b = (-2/len(data)) * np.sum(data[:, 1] - (m * data[:, 0] + b))

        m -= learn_rate * derivative_m
        b -= learn_rate * derivative_b

    return m,b

#run gradient descent
slope, intercept = gradient_descent(data_points)

#create scatter plot of data points & overlay fitted line
plt.scatter(data_points[:, 0], data_points[:, 1], label='Data Points')
plt.plot(data_points[:, 0], slope * data_points[:, 0] + intercept, color='blue', label='Gradient Descent Fitted Line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Gradient Descent Project Best Line of Fit')
plt.legend()
plt.show()

#linear regression model - fit it to data points to get slope and int
model= LinearRegression()
model.fit(data_points[:, 0].reshape(-1,1), data_points[:, 1])
sklearn_slope, sklearn_intercept = model.coef_[0], model.intercept_

#results
print("Gradient Descent Result:")
print(f"Slope = {slope}, Intercept = {intercept}")

print("\nSklearn Linear Regression Result:")
print(f"Slope = {sklearn_slope}, Intercept = {sklearn_intercept}")
