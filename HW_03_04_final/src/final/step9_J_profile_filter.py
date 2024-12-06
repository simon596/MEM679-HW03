import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data from the CSV file into a DataFrame
data = pd.read_csv(r'.\results\profile.csv')
data.head()  # Display the first few rows of the dataset for verification

# Adjust the second column (named '1.0' here) by subtracting a constant
# This operation is performed to shift the data so that it starts at 1.0
data['1.0'] -= 0.0328969474598937

# Plot the adjusted data as a scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(data['0.0'], data['1.0'], color='blue', label='Data Points')
plt.title('Scatter Plot of Adjusted Data')
plt.xlabel('Column 1')
plt.ylabel('Adjusted Column 2')
plt.legend()
plt.grid(True)
plt.show()

# Prepare the data for linear regression
# Reshape is required to convert the series into a 2D array as expected by scikit-learn
X = data['0.0'].values.reshape(-1, 1)
y = data['1.0'].values

# Create and fit a linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)  # Predict the y values for the given X

# Plot the data points and the regression line
plt.figure(figsize=(10, 5))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.scatter(data['0.0'], data['1.0'], color='blue', label='Data Points')
plt.plot(data['0.0'], y_pred, color='red', lw=4, label='Regression Filter')
plt.xlabel('Compressive strain (mm/mm)', fontsize=22)
plt.ylabel('Determinant of deformation\ngradient ($mm^3$/$mm^3$)', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True)
plt.show()

# Print the linear regression coefficients (slope and intercept)
slope = model.coef_[0]
intercept = model.intercept_
(slope, intercept)  # Display the slope and intercept
