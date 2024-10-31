# First, install the necessary libraries
# Run this in your terminal or Jupyter Notebook cell
# !pip install pandas matplotlib seaborn

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import panel as pn
import hvplot.pandas

pn.extension()

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


# Create widgets for interactive selection
stiffness_select = pn.widgets.Select(
    name='Relative Stiffness', 
    options=dataset['Relative Stiffness'].unique().tolist()
)
concentration_slider = pn.widgets.FloatSlider(
    name='Particle Concentration', 
    start=dataset['Particle Concentration'].min(), 
    end=dataset['Particle Concentration'].max(), 
    step=0.1
)

# Define the plotting function, which will filter the dataset based on widget values
@pn.depends(
    stiffness=stiffness_select.param.value, 
    concentration=concentration_slider.param.value
)
def violin_plot(stiffness, concentration):
    # Filter dataset based on widget selections
    filtered_data = dataset[
        (dataset['Relative Stiffness'] == stiffness) & 
        (dataset['Particle Concentration'] == concentration)
    ]
    
    # Plot violin plot
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x='Relative Stiffness', 
        y='Est. Volume Fraction ', 
        data=filtered_data
    )
    plt.title(f'Violin Plot of Estimated Volume Fraction\n'
              f'Relative Stiffness: {stiffness}, Particle Concentration: {concentration}')
    plt.xlabel('Relative Stiffness')
    plt.ylabel('Estimated Volume Fraction')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

# Create the interactive layout
interactive_panel = pn.Column(
    "### Interactive Violin Plot of Estimated Volume Fraction",
    "Adjust the sliders to filter data based on Relative Stiffness and Particle Concentration:",
    pn.Row(stiffness_select, concentration_slider),
    violin_plot
)

# To display in a Jupyter notebook, use:
interactive_panel.show()

# For standalone use in a browser, save this code to a file (e.g., `app.py`) and run:
# panel serve app.py