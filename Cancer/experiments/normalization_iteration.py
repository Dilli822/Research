import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestCentroid

# Load the breast cancer dataset from sklearn
cancer_data = load_breast_cancer()

# Convert the dataset to a pandas DataFrame for easier manipulation
custom_data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
labels = cancer_data.target  # Labels for the dataset

# Apply MinMaxScaler for normalization
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(custom_data), columns=custom_data.columns)

# Check if the dataset contains negative values
min_value = custom_data.min().min()
if min_value < 0:
    data_shifted = custom_data + abs(min_value) + 1  # Shift all data to positive range
else:
    data_shifted = custom_data + 1  # If no negative values, just add 1

# Normalize the shifted data
max_values = data_shifted.max(axis=0)
CUSTOM_normalized = data_shifted / max_values

# Add new classifiers to the models dictionary
models = {
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Ridge Classifier': RidgeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Multi-layer Perceptron': MLPClassifier(max_iter=1000),
    'Stochastic Gradient Descent': SGDClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Perceptron': Perceptron(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Passive Aggressive': PassiveAggressiveClassifier(),
    'Histogram-based Gradient Boosting': HistGradientBoostingClassifier(),
    'Voting Classifier': VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gnb', GaussianNB())
    ]),
    'Calibrated Classifier': CalibratedClassifierCV(RandomForestClassifier(), method='sigmoid'),
    'Nearest Centroid': NearestCentroid()
}


# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, models):
    accuracies = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[model_name] = accuracy_score(y_test, y_pred)
    return accuracies

# Number of repetitions
n_reps = 1
better_manual_count = {model_name: 0 for model_name in models.keys()}

# Run multiple iterations
for _ in range(n_reps):
    # Split data into training and testing sets using normalized data (MinMaxScaler)
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)
    normalized_accuracies = evaluate_models(X_train, X_test, y_train, y_test, models)

    # Split data into training and testing sets using manually normalized data
    X_train, X_test, y_train, y_test = train_test_split(CUSTOM_normalized, labels, test_size=0.2, random_state=42)
    manual_accuracies = evaluate_models(X_train, X_test, y_train, y_test, models)

    # Compare results and count where manually normalized data worked better
    for model_name in models.keys():
        if manual_accuracies[model_name] > normalized_accuracies[model_name]:
            better_manual_count[model_name] += 1

# Prepare data for tabulate with added accuracy columns and difference column
table_data = [
    [
        model_name, 
        better_manual_count[model_name], 
        n_reps - better_manual_count[model_name], 
        round(manual_accuracies[model_name], 2),  # Rounded to 3 decimal places
        round(normalized_accuracies[model_name], 2),  # Rounded to 3 decimal places
        round(manual_accuracies[model_name] - normalized_accuracies[model_name], 2)  # Accuracy difference rounded
    ]
    for model_name in models.keys()
]

headers = [
    "Model", 
    "Manual Performed Better", 
    "MinMax Performed Better", 
    "Manual Accuracy", 
    "MinMax Accuracy", 
    "Accuracy Difference"
]

# Print results as a table
print(tabulate(table_data, headers=headers, tablefmt="grid"))
