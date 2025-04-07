import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the breast cancer dataset from sklearn
cancer_data = load_breast_cancer()

# Convert the dataset to a pandas DataFrame for easier manipulation
custom_data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
labels = cancer_data.target  # Labels for the dataset

# Apply MinMaxScaler for normalization
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(custom_data), columns=custom_data.columns)

# Add 1 to the original data and normalize by max value
data_plus_one = custom_data + 1
max_values = data_plus_one.max(axis=0)
data_normalized = data_plus_one / max_values

# Split data into training and testing sets using normalized data (MinMaxScaler)
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)

# Train a model (for example, SVM)
model = SVC()
model.fit(X_train, y_train)

# Predict on the test data and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using normalized data (MinMaxScaler): {accuracy:.4f}")

# Now, split the data using the manually normalized data
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)

# Train a model (for example, RandomForest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test data and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using manually normalized data: {accuracy:.4f}")
