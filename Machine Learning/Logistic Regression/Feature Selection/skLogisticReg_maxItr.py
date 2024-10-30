

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create a more complex synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=15, random_state=42)

# Create and train the logistic regression model without max_iter (default 100)
model_no_max_iter = LogisticRegression()
model_no_max_iter.fit(X, y)

# Coefficients without max_iter
no_max_iter_coefficients = model_no_max_iter.coef_[0]

# Create and train the logistic regression model with low max_iter
model_low_max_iter = LogisticRegression(max_iter=10)  # Setting max_iter to 10
model_low_max_iter.fit(X, y)

# Coefficients with low max_iter
low_max_iter_coefficients = model_low_max_iter.coef_[0]

# Create and train the logistic regression model with high max_iter
model_high_max_iter = LogisticRegression(max_iter=1500)  # Setting max_iter to 500
model_high_max_iter.fit(X, y)

# Coefficients with high max_iter
high_max_iter_coefficients = model_high_max_iter.coef_[0]

# Visualization of the coefficients
plt.figure(figsize=(18, 6))

# Coefficients without max_iter
plt.subplot(1, 3, 1)
plt.bar(range(len(no_max_iter_coefficients)), no_max_iter_coefficients, color='orange')
plt.title('Logistic Regression Coefficients (default max_iter=100)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Coefficients with low max_iter
plt.subplot(1, 3, 2)
plt.bar(range(len(low_max_iter_coefficients)), low_max_iter_coefficients, color='red')
plt.title('Logistic Regression Coefficients (max_iter=10)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Coefficients with high max_iter
plt.subplot(1, 3, 3)
plt.bar(range(len(high_max_iter_coefficients)), high_max_iter_coefficients, color='blue')
plt.title('Logistic Regression Coefficients (max_iter=500)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

plt.tight_layout()
plt.show()



