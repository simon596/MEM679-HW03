# First, install the necessary libraries
# Run this in your terminal or Jupyter Notebook cell
# !pip install pandas matplotlib seaborn

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import panel as pn
import hvplot.pandas

# Load the dataset
file_path = './dataset/Dataset_PhaseSepVolumeFractionEstimation.csv'  # Relative path from root directory
dataset = pd.read_csv(file_path)

#%% heat map for correlation matrix
# Calculate the correlation matrix for numeric columns
correlation_matrix = dataset.select_dtypes(include=['float64', 'int64']).corr()
# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap of Hydrogel Dataset Features")
plt.show()

#%% Create a violin plot of Estimated Volume Fraction based on Relative Stiffness
plt.figure(figsize=(10, 6))
sns.violinplot(x='Relative Stiffness', y='Est. Volume Fraction ', data=dataset)
plt.xlabel('Relative Stiffness')
plt.ylabel('Estimated Volume Fraction')
plt.title('Violin Plot of Estimated Volume Fraction by Relative Stiffness')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#%% interactive panel
pn.extension()

# Create a slider widget for "Vol of Particles After Settling (mL)"
volume_slider = pn.widgets.FloatSlider(
    name='Vol of Particles After Settling (mL)', 
    start=dataset['Vol of Particles After Settling (mL)'].min(), 
    end=dataset['Vol of Particles After Settling (mL)'].max(), 
    step=0.5
)

# Define the plotting function
@pn.depends(volume=volume_slider.param.value)
def histogram_plot(volume):
    # Filter dataset based on the slider value
    filtered_data = dataset[dataset['Vol of Particles After Settling (mL)'] >= volume]
    
    # Check if filtered_data is empty
    if filtered_data.empty:
        return "No data available for the selected volume."

    # Plot histogram of "Relative Stiffness"
    plt.figure(figsize=(8, 5))
    filtered_data['Relative Stiffness'].value_counts().plot(kind='bar')
    plt.title(f'Histogram of Relative Stiffness\n(Vol of Particles After Settling >= {volume} mL)')
    plt.xlabel('Relative Stiffness')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

# Create the interactive layout
interactive_panel = pn.Column(
    "### Interactive Histogram of Relative Stiffness",
    "Adjust the slider to filter data based on 'Vol of Particles After Settling (mL)':",
    volume_slider,
    histogram_plot  # Include the function directly
)

# To display in a Jupyter notebook, use:
interactive_panel.show()

# For standalone use in a browser, save this code to a file (e.g., `app.py`) and run:
# panel serve app.py