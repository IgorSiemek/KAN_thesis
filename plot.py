import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Provided summary of metrics
data = {
    "Model": ["MLP", "KAN"] * 11,
    "MSE": [
        3.511570e-02, 6.707572e-04,
        3.551073e-02, 1.029898e-03,
        3.848636e-02, 1.680877e-03,
        5.027150e-02, 4.617541e-03,
        4.843366e-02, 2.217394e-03,
        2.825906e-03, 9.984574e-04,
        2.681494e-04, 8.981982e-04,
        6.822738e-04, 7.646478e-04,
        5.559645e-05, 3.756134e-04,
        3.657003e-05, 5.213107e-04,
        6.195950e-04, 5.593712e-04
    ]
}

df = pd.DataFrame(data)

accuracy = [82.74, 86.49]  # First dataset
accuracy2 = [89.66, 96.67]  # Second dataset

# Calculate mean RMSE for KAN and MLP (RMSE = sqrt(MSE))
mean_rmse_kan = np.sqrt(df[df["Model"] == "KAN"]["MSE"]).mean()
mean_rmse_mlp = np.sqrt(df[df["Model"] == "MLP"]["MSE"]).mean()

# Mean accuracy values (update these with correct values if needed)

mean_accuracy_kan = np.mean([accuracy[0], accuracy2[0]])
mean_accuracy_mlp = np.mean([accuracy[1], accuracy2[1]])

# Define models for plotting
models = ["KAN", "MLP"]

# Update RMSE plot with calculated values
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot RMSE with scientific notation
axes[0].bar(models, [mean_rmse_kan, mean_rmse_mlp], color=["#f4a7a3", "#74b9ff"])
axes[0].set_ylabel("RMSE")
axes[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[0].set_title("Model RMSE Comparison")

# Plot Accuracy comparison
axes[1].bar(models, [mean_accuracy_kan, mean_accuracy_mlp], color=["#f4a7a3", "#74b9ff"])
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(80, 100)
axes[1].set_title("Model Accuracy Comparison")

plt.tight_layout()
plt.show()
