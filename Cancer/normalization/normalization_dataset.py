import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml

# List of all classifiers to evaluate
classifiers = {
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Ridge Classifier': RidgeClassifier(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'MLP Classifier': MLPClassifier(max_iter=1000)  # Multi-layer Perceptron
}

# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, classifiers, dataset_name):
    accuracies = {}
    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[classifier_name] = accuracy
    return accuracies

# Function to fetch datasets from OpenML and sklearn
def load_sklearn_datasets():
    datasets = {}
    
    # Load a selection of datasets from sklearn.datasets
    from sklearn.datasets import  load_digits, load_iris, load_wine, load_breast_cancer, load_diabetes
    datasets = {
   
        "Iris": load_iris(),
        "Wine": load_wine(),
        "Breast Cancer": load_breast_cancer(),
        "Diabetes": load_diabetes(),
    }

    return datasets

datasets = load_sklearn_datasets()

# Number of repetitions
n_reps = 10

# To store results across all repetitions
normalized_accuracies_list = []
manual_accuracies_list = []
better_model_count = {name: 0 for name in classifiers.keys()}

# Run the experiment 1000 times for each dataset
for dataset_name, dataset in datasets.items():
    print(f"\nRunning experiment on {dataset_name} dataset...")
    
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    labels = dataset.target  # Labels for the dataset

    # Apply MinMaxScaler for normalization
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Add 1 to the original data and normalize by max value
    data_plus_one = data + 1
    max_values = data_plus_one.max(axis=0)
    data_normalized = data_plus_one / max_values

    # To store results across all repetitions for the current dataset
    normalized_accuracies_list = []
    manual_accuracies_list = []
    better_model_count = {name: 0 for name in classifiers.keys()}
    
    for _ in range(n_reps):
        # Split data into training and testing sets using normalized data (MinMaxScaler)
        X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)
        normalized_accuracies = evaluate_models(X_train, X_test, y_train, y_test, classifiers, dataset_name)
        normalized_accuracies_list.append(normalized_accuracies)

        # Now, split the data using manually normalized data
        X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)
        manual_accuracies = evaluate_models(X_train, X_test, y_train, y_test, classifiers, dataset_name)
        manual_accuracies_list.append(manual_accuracies)

        # Compare accuracies and count where manually normalized data outperforms
        for classifier_name in classifiers.keys():
            if manual_accuracies[classifier_name] > normalized_accuracies[classifier_name]:
                better_model_count[classifier_name] += 1

    # Calculate the average improvement
    print("Results after 1000 repetitions:")
    for classifier_name in classifiers.keys():
        improvement_percentage = (better_model_count[classifier_name] / n_reps) * 100
        print(f"{classifier_name} improved with manually normalized data {improvement_percentage:.2f}% of the time")

