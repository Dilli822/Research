import numpy as np

# Dataset
# Hours = np.array( [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50] )
# Pass = np.array( [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1] )


Hours = np.array( [1,2] )
Pass = np.array( [0, 1] )

# Parameters
alpha = 0.1  # Learning rate
threshold = 1e-6  # Convergence threshold
iterations = 3 # Maximum number of iterations

# Initialize coefficients
beta_0 = 0.0
beta_1 = 0.0

for i in range(iterations):
    # Compute predicted probabilities
    linear_combination = beta_0 + beta_1 * Hours
    p = 1 / (1 + np.exp(-linear_combination))
    
    # Calculate gradients
    gradient_0 = np.sum(p - Pass) / len(Pass)  # Average gradient
    gradient_1 = np.sum((p - Pass) * Hours) / len(Pass)  # Average gradient

    # Update coefficients
    beta_0_new = beta_0 - alpha * gradient_0
    beta_1_new = beta_1 - alpha * gradient_1
    
    # Check for convergence
    if abs(beta_0_new - beta_0) < threshold and abs(beta_1_new - beta_1) < threshold:
        break
    
    beta_0, beta_1 = beta_0_new, beta_1_new

print(f"Converged after {i + 1} iterations")
print(f"Beta_0 (Intercept): {beta_0:.6f}")
print(f"Beta_1 (Coefficient for Hours): {beta_1:.6f}")
