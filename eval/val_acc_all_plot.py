import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Read the CSV file
filename = "data/val_acc_all_models_results.csv"
df = pd.read_csv(filename)

# Use a clean style without grid by default
plt.style.use("seaborn-v0_8-white")

# Set global font settings
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

# Define the columns for X and Y axes
x_col = "Step"
y_cols = [
    "sopas_fpr0.3 - val/acc",
    # "sopas_fpr0.5 - val/acc",
    "bce - val/acc",
    "sotas_tpr0.6_fpr0.4 - val/acc",
    # "sotas_tpr0.5_fpr0.5 - val/acc",
]

max_step = df[x_col].max()
max_bce_epoch = max_step / 1730
# Create the figure with high DPI
plt.figure(figsize=(5, 4), dpi=300)

# Define a color palette
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Iterate through the columns and plot each one
for i, y_col in enumerate(y_cols):
    # Determine the label and the specific step-per-epoch ratio
    if "sopas" in y_col:
        label_name = "SOPA-s"
        steps_per_epoch = 1880 # SOPAs conversion
    if "sotas" in y_col:
        label_name = "SOTA-s"
        steps_per_epoch = 1880 # SOTAs conversion
    if "bce" in y_col:
        label_name = "BCE"
        steps_per_epoch = 1730 # BCE conversion

    # Calculate the specific Epochs x-axis for this line
    # (Step / steps_per_epoch)
    x_data_epochs = df[x_col] / steps_per_epoch

    # Plot the line using the calculated epochs as x-axis
    plt.plot(x_data_epochs, df[y_col], label=label_name, linewidth=3, color=colors[i % len(colors)])

# Set axis labels and title
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.title("Validation Accuracy", fontsize=18)

ticks = np.arange(0, int(max_bce_epoch) + 1)
plt.xticks(ticks)
plt.xlim(0, max_bce_epoch * 1.02)
plt.tick_params(axis='both', which='major', labelsize=14)

# Configure the legend
plt.legend(frameon=True, fontsize=14, loc="lower right")

# Explicitly disable the grid
plt.grid(False)

# Remove the top and right spines
sns.despine()

# Adjust layout
plt.tight_layout()

output_dir = "figures"
output_filename = "val_acc_all_models_comparison.pdf"
output_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)
print(f"Saving figure to {output_path}...")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("Done.")
# Show the plot
plt.show()
