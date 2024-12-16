import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# Apply StandardScaler for standardization
standardizer = StandardScaler()
standardized_data = pd.DataFrame(standardizer.fit_transform(custom_data), columns=custom_data.columns)

# Add 1 to the original data and normalize by max value
data_plus_one = custom_data + 1
max_values = custom_data.max(axis=0)  # Use the original max values (no +1 added to max)
CUSTOM_normalized = data_plus_one / max_values

# List of models to evaluate
models = {
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Function to train and evaluate models
def evaluate_model(X_train, X_test, y_train, y_test, models, dataset_name):
    print(f"Results for {dataset_name}:")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
    print()

# Split data into training and testing sets using normalized data (MinMaxScaler)
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)
evaluate_model(X_train, X_test, y_train, y_test, models, "Normalized Data (MinMaxScaler)")

# Split data into training and testing sets using standardized data (StandardScaler)
X_train, X_test, y_train, y_test = train_test_split(standardized_data, labels, test_size=0.2, random_state=42)
evaluate_model(X_train, X_test, y_train, y_test, models, "Standardized Data (StandardScaler)")

# Split data into training and testing sets using manually normalized data
X_train, X_test, y_train, y_test = train_test_split(CUSTOM_normalized, labels, test_size=0.2, random_state=42)
evaluate_model(X_train, X_test, y_train, y_test, models, "Manually Normalized Data")
