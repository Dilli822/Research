# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import RidgeClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neural_network import MLPClassifier

# # Load the breast cancer dataset from sklearn
# cancer_data = load_breast_cancer()

# # Convert the dataset to a pandas DataFrame for easier manipulation
# custom_data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
# labels = cancer_data.target  # Labels for the dataset

# # Apply MinMaxScaler for normalization
# scaler = MinMaxScaler()
# normalized_data = pd.DataFrame(scaler.fit_transform(custom_data), columns=custom_data.columns)

# # Add 1 to the original data and normalize by max value
# data_plus_one = custom_data + 1
# max_values = data_plus_one.max(axis=0)
# data_normalized = data_plus_one / max_values

# # List of all classifiers to evaluate
# classifiers = {
#     'SVM': SVC(),
#     'Logistic Regression': LogisticRegression(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'AdaBoost': AdaBoostClassifier(),
#     'Bagging': BaggingClassifier(),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Naive Bayes': GaussianNB(),
#     'Ridge Classifier': RidgeClassifier(),
#     'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
#     'MLP Classifier': MLPClassifier(max_iter=1000)  # Multi-layer Perceptron
# }

# # Function to train and evaluate models
# def evaluate_models(X_train, X_test, y_train, y_test, classifiers, dataset_name):
#     print(f"Results for {dataset_name}:")
#     for classifier_name, classifier in classifiers.items():
#         classifier.fit(X_train, y_train)
#         y_pred = classifier.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"{classifier_name} Accuracy: {accuracy:.4f}")
#     print()

# # Split data into training and testing sets using normalized data (MinMaxScaler)
# X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)
# evaluate_models(X_train, X_test, y_train, y_test, classifiers, "Normalized Data (MinMaxScaler)")

# # Now, split the data using manually normalized data
# X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)
# evaluate_models(X_train, X_test, y_train, y_test, classifiers, "Manually Normalized Data")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import RidgeClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neural_network import MLPClassifier

# # Load the breast cancer dataset from sklearn
# cancer_data = load_breast_cancer()

# # Convert the dataset to a pandas DataFrame for easier manipulation
# custom_data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
# labels = cancer_data.target  # Labels for the dataset

# # Apply MinMaxScaler for normalization
# scaler = MinMaxScaler()
# normalized_data = pd.DataFrame(scaler.fit_transform(custom_data), columns=custom_data.columns)

# # Add 1 to the original data and normalize by max value
# data_plus_one = custom_data + 1
# max_values = data_plus_one.max(axis=0)
# data_normalized = data_plus_one / max_values

# # List of all classifiers to evaluate
# classifiers = {
#     'SVM': SVC(),
#     'Logistic Regression': LogisticRegression(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'AdaBoost': AdaBoostClassifier(),
#     'Bagging': BaggingClassifier(),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Naive Bayes': GaussianNB(),
#     'Ridge Classifier': RidgeClassifier(),
#     'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
#     'MLP Classifier': MLPClassifier(max_iter=1000)  # Multi-layer Perceptron
# }

# # Function to train and evaluate models
# def evaluate_models(X_train, X_test, y_train, y_test, classifiers, dataset_name):
#     print(f"Results for {dataset_name}:")
#     accuracies = {}
#     for classifier_name, classifier in classifiers.items():
#         classifier.fit(X_train, y_train)
#         y_pred = classifier.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         accuracies[classifier_name] = accuracy
#         print(f"{classifier_name} Accuracy: {accuracy:.4f}")
#     return accuracies

# # Split data into training and testing sets using normalized data (MinMaxScaler)
# X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)
# normalized_accuracies = evaluate_models(X_train, X_test, y_train, y_test, classifiers, "Normalized Data (MinMaxScaler)")

# # Now, split the data using manually normalized data
# X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)
# manual_accuracies = evaluate_models(X_train, X_test, y_train, y_test, classifiers, "Manually Normalized Data")

# # Compare accuracies and print where manually normalized data outperforms
# print("\nComparison of Accuracies:")
# for classifier_name in classifiers.keys():
#     if manual_accuracies[classifier_name] > normalized_accuracies[classifier_name]:
#         print(f"\nIn the {classifier_name} model, manually normalized data performed better:")
#         print(f"Manually Normalized Accuracy: {manual_accuracies[classifier_name]:.4f} | Normalized Accuracy: {normalized_accuracies[classifier_name]:.4f}")



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
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

# Number of repetitions
n_reps = 100

# To store results across all repetitions
normalized_accuracies_list = []
manual_accuracies_list = []
better_model_count = {name: 0 for name in classifiers.keys()}

# Run the experiment 1000 times
for _ in range(n_reps):
    # Split data into training and testing sets using normalized data (MinMaxScaler)
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)
    normalized_accuracies = evaluate_models(X_train, X_test, y_train, y_test, classifiers, "Normalized Data (MinMaxScaler)")
    normalized_accuracies_list.append(normalized_accuracies)

    # Now, split the data using manually normalized data
    X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)
    manual_accuracies = evaluate_models(X_train, X_test, y_train, y_test, classifiers, "Manually Normalized Data")
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
