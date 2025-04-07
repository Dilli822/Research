# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tabulate import tabulate
# from itertools import combinations

# # Load dataset
# data = load_breast_cancer()
# X, y = data.data, data.target

# # Convert to DataFrame for easier handling
# X_df = pd.DataFrame(X, columns=data.feature_names)

# # Wrapper-based feature selection
# def wrapper_feature_selection(X, y, classifier, eval_metric="accuracy", max_features=None):
#     """
#     Perform wrapper feature selection using the given classifier and evaluation metric.
#     """
#     if max_features is None:
#         max_features = X.shape[1]
    
#     best_score = 0
#     best_features = None
    
#     for num_features in range(1, max_features + 1):
#         for feature_subset in combinations(X.columns, num_features):
#             X_subset = X[list(feature_subset)]
#             scores = cross_val_score(classifier, X_subset, y, cv=5, scoring=eval_metric)
#             mean_score = np.mean(scores)
            
#             if mean_score > best_score:
#                 best_score = mean_score
#                 best_features = feature_subset
    
#     return best_features, best_score

# # Classifiers
# classifiers = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "Support Vector Machine": SVC(random_state=42),
#     "K-Nearest Neighbors": KNeighborsClassifier(),
#     "Naive Bayes": GaussianNB(),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#     "AdaBoost": AdaBoostClassifier(random_state=42),
# }

# # Evaluate each classifier for wrapper-based feature selection
# results = []
# for clf_name, clf in classifiers.items():
#     selected_features, best_score = wrapper_feature_selection(X_df, y, clf, eval_metric="accuracy", max_features=5)
#     results.append([clf_name, ", ".join(selected_features), best_score])

# # Display Results
# headers = ["Classifier", "Selected Features", "Best Accuracy"]
# print(tabulate(results, headers=headers, tablefmt="grid"))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from itertools import combinations
from tabulate import tabulate

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert to DataFrame for easier handling
X_df = pd.DataFrame(X, columns=data.feature_names)

# Wrapper-based feature selection
def wrapper_feature_selection(X, y, classifier, eval_metric="accuracy", max_features=None):
    """
    Perform wrapper feature selection using the given classifier and evaluation metric.
    """
    if max_features is None:
        max_features = X.shape[1]
    
    best_score = 0
    best_features = None
    
    for num_features in range(1, max_features + 1):
        for feature_subset in combinations(X.columns, num_features):
            X_subset = X[list(feature_subset)]
            scores = cross_val_score(classifier, X_subset, y, cv=5, scoring=eval_metric)
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_features = feature_subset
    
    return best_features, best_score

# Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

# Evaluate each classifier for wrapper-based feature selection
results = []
for clf_name, clf in classifiers.items():
    selected_features, best_score = wrapper_feature_selection(X_df, y, clf, eval_metric="accuracy", max_features=5)
    results.append([clf_name, ", ".join(selected_features), round(best_score, 4)])

# Convert results to DataFrame for better handling
df_results = pd.DataFrame(results, columns=["Classifier", "Selected Features", "Best Accuracy"])

# Function to save the results as a PNG
def save_results_to_png(df, filename):
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.8 + 1))  # Adjust figure size based on the number of rows
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the table as a PNG
    plt.title("Wrapper-Based Feature Selection Results", fontsize=14, pad=20)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Save the table to a PNG file
save_results_to_png(df_results, "wrapper_results.png")
print("Results saved as 'wrapper_results.png'.")
