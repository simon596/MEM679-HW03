# First, install the necessary libraries
# Run this in your terminal or Jupyter Notebook cell
# !pip install pandas matplotlib seaborn

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../dataset/Dataset_PhaseSepVolumeFractionEstimation.csv'  # Replace with your file path
dataset = pd.read_csv(file_path)

# Calculate the correlation matrix for numeric columns
correlation_matrix = dataset.select_dtypes(include=['float64', 'int64']).corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap of Dataset Features")
plt.show()
