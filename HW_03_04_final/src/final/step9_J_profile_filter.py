import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the spreadsheet to check its contents
data = pd.read_csv(r'.\results\profile.csv')
data.head()

# Subtract 0.024862 from each element in the second column
# in order to make it start at '1.0'
data['1.0'] -= 0.0328969474598937

# Plotting the modified data
plt.figure(figsize=(10, 5))
plt.scatter(data['0.0'], data['1.0'], color='blue', label='Data Points')
plt.title('Scatter Plot of Adjusted Data')
plt.xlabel('Column 1')
plt.ylabel('Adjusted Column 2')
plt.legend()
plt.grid(True)
plt.show()

# Performing linear regression
X = data['0.0'].values.reshape(-1, 1)
y = data['1.0'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

#%% Adding the regression line to the plot
plt.figure(figsize=(10, 5))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.scatter(data['0.0'], data['1.0'], color='blue', label='Data Points')
plt.plot(data['0.0'], y_pred, color='red', lw=4, label='Regression Filter')
#plt.title('Volume Change Profile', fontsize=24)
plt.xlabel('Compressive strain (mm/mm)', fontsize=22)
plt.ylabel('Determinant of deformation\ngradient ($mm^3$/$mm^3$)', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True)
#plt.text(0.03,0.90, '$det(J)|_{\epsilon=0.25}=0.917$', fontsize=22)
plt.show()

# Regression coefficients
slope = model.coef_[0]
intercept = model.intercept_
(slope, intercept)
# %%
