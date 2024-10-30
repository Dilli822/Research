import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Generate random data points
np.random.seed(42)  # For reproducibility
num_points = 100
data_x = np.random.uniform(0, 10, size=num_points)

# Create figure and axis
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121)  # 2D histogram
ax3d = fig.add_subplot(122, projection='3d')  # 3D scatter

# <div contenteditable="plaintext-only"></div>
# Mean and Standard Deviation formulas
mean_formula = r'mean $\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$'
std_dev_formula = r'standard deviation $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$'

# Function to update the animation
def update(frame):
    ax.clear()  # Clear the previous plot
    ax3d.clear()  # Clear the previous 3D plot
    
    # Draw the histogram
    ax.hist(data_x, bins=30, color='lightgray', alpha=0.7, edgecolor='black')

    # Recalculate mean and standard deviation for the current frame
    current_points = data_x[:frame * (num_points // 100)]
    mean_value = np.mean(current_points)
    std_dev_value = np.std(current_points)
    
    # Draw the mean line
    ax.axvline(mean_value, color='yellow', linestyle='--', linewidth=2, label='Mean')

    # Draw standard deviation lines
    ax.axvline(mean_value + std_dev_value, color='orange', linestyle='--', linewidth=2, label='Mean + 1 Std Dev')
    ax.axvline(mean_value - std_dev_value, color='orange', linestyle='--', linewidth=2, label='Mean - 1 Std Dev')
    
    # Set limits and labels based on current data
    ax.set_xlim(-15, 15)  # Update limits to accommodate the range
    ax.set_ylim(0, 150)
    ax.set_title('Distribution of Random Data Points with Mean and Standard Deviation')
    ax.set_xlabel('Data Value')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')

    # Display the formulas on the plot
    ax.text(0.05, 0.95, mean_formula, fontsize=14, ha='left', va='top', transform=ax.transAxes)
    ax.text(0.05, 1.15, std_dev_formula, fontsize=14, ha='left', va='top', transform=ax.transAxes)

    # 3D Scatter Plot
    x = np.arange(len(current_points))
    y = current_points
    z = np.zeros_like(current_points)  # Z coordinates are zero for a flat representation

    ax3d.scatter(x, y, z, color='blue', alpha=0.5)
    
    # Draw mean and standard deviation in 3D
    ax3d.scatter([x.mean()], [mean_value], [0], color='yellow', s=100, label='Mean', marker='o')
    ax3d.scatter([x.mean() - std_dev_value], [mean_value - std_dev_value], [0], color='orange', s=100, label='Mean - 1 Std Dev', marker='o')
    ax3d.scatter([x.mean() + std_dev_value], [mean_value + std_dev_value], [0], color='orange', s=100, label='Mean + 1 Std Dev', marker='o')

    # Set labels and title for 3D plot
    ax3d.set_xlabel('Index')
    ax3d.set_ylabel('Data Value')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Scatter Plot of Random Data Points')
    ax3d.legend(loc='upper right')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=np.arange(1, 100), repeat=False)

# Display the animation
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [-2, -4, 6, 8, 10]

# Calculate the Pearson correlation coefficient
mean_x = np.mean(x)
mean_y = np.mean(y)
cov_xy = np.sum((x - mean_x) * (y - mean_y)) / (len(x) - 1)
std_x = np.std(x)
std_y = np.std(y)
r = cov_xy / (std_x * std_y)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points', alpha=0.6)

# Add a trend line
m, b = np.polyfit(x, y, 1)  # Linear regression
plt.plot(x, m * np.array(x) + b, color='orange', label='Trend Line')

# Set titles and labels
plt.title(f'Scatter Plot with Pearson Correlation Coefficient: {r:.2f}', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)

# Add a grid and legend
plt.grid()
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
x = np.random.uniform(0, 10, 100)  # Dataset X
y = 2 * x + np.random.normal(0, 2, 100)  # Dataset Y with positive covariance

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for datasets X and Y
axs[0].scatter(x, y, color='blue', alpha=0.6)
axs[0].set_title('Scatter Plot of X vs Y', fontsize=14)
axs[0].set_xlabel('X', fontsize=12)
axs[0].set_ylabel('Y', fontsize=12)
axs[0].grid(True)

# Add covariance and variance formulas
formula_str = (
    r"$\text{Variance (X)} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu_X)^2$" + "\n" +
    r"$\text{Variance (Y)} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \mu_Y)^2$" + "\n" +
    r"$\text{Covariance (X, Y)} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu_X)(y_i - \mu_Y)$"
)

axs[1].text(0.1, 0.5, formula_str, fontsize=14, verticalalignment='center')
axs[1].set_title('Mathematical Formulas', fontsize=14)
axs[1].axis('off')  # Hide axes for the formula plot

plt.tight_layout()
plt.show()
