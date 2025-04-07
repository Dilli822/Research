# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import Pipeline
# from sklearn.base import clone
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier

# # Load and preprocess dataset
# data = load_breast_cancer()
# X, y = data.data, data.target

# # Scale data for Chi-Square and Mutual Information
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# # Filter Methods
# def apply_filter_methods(X, y):
#     results = {}

#     # 1. Variance Threshold
#     vt = VarianceThreshold(threshold=0.01)
#     X_vt = vt.fit_transform(X)
#     results['Variance Threshold'] = X_vt

#     # 2. Correlation
#     corr_matrix = np.corrcoef(X.T, y, rowvar=True)
#     corr_threshold = 0.2
#     selected_features_corr = [i for i in range(X.shape[1]) if abs(corr_matrix[i, -1]) > corr_threshold]
#     X_corr = X[:, selected_features_corr]
#     results['Correlation'] = X_corr

#     # 3. Chi-Square
#     chi2_selector = SelectKBest(score_func=chi2, k=10)
#     X_chi2 = chi2_selector.fit_transform(X_scaled, y)
#     results['Chi-Square'] = X_chi2

#     # 4. Mutual Information
#     mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
#     X_mi = mi_selector.fit_transform(X_scaled, y)
#     results['Mutual Information'] = X_mi

#     return results

# # Classifiers
# classifiers = {
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "Support Vector Machine": SVC(random_state=42),
#     "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "K-Nearest Neighbors": KNeighborsClassifier(),
#     "Naive Bayes": GaussianNB(),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#     "AdaBoost": AdaBoostClassifier(random_state=42),
# }

# # Evaluate Models
# def evaluate_models(X_train, X_test, y_train, y_test, classifiers):
#     results = []
#     for name, clf in classifiers.items():
#         model = clone(clf)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)

#         results.append({
#             "Classifier": name,
#             "Accuracy": accuracy,
#             "Precision": precision,
#             "Recall": recall,
#             "F1 Score": f1
#         })
#     return results

# # Main Execution
# filter_results = apply_filter_methods(X, y)

# final_results = {}
# for filter_name, X_filtered in filter_results.items():
#     print(f"\nApplying Filter: {filter_name}")
#     X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=42)

#     results = evaluate_models(X_train, X_test, y_train, y_test, classifiers)
#     final_results[filter_name] = results

#     for result in results:
#         print(f"Classifier: {result['Classifier']}, Accuracy: {result['Accuracy']:.2f}, "
#               f"Precision: {result['Precision']:.2f}, Recall: {result['Recall']:.2f}, F1 Score: {result['F1 Score']:.2f}")

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Convert results into a DataFrame for each filter
# all_results = []
# for filter_name, results in final_results.items():
#     for result in results:
#         result['Filter'] = filter_name
#         all_results.append(result)

# df_results = pd.DataFrame(all_results)

# # Save results as a PNG image
# def save_results_to_png(df, filename):
#     plt.figure(figsize=(15, 8))
#     sns.set(style="whitegrid")
#     sns.barplot(
#         data=df.melt(id_vars=["Classifier", "Filter"], 
#                      value_vars=["Accuracy", "Precision", "Recall", "F1 Score"], 
#                      var_name="Metric", 
#                      value_name="Score"),
#         x="Metric", 
#         y="Score", 
#         hue="Classifier"
#     )
#     plt.title("Performance of Classifiers by Metric and Filter")
#     plt.xticks(rotation=45)
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()

# save_results_to_png(df_results, "filter_classifier_performance.png")
# print("Results saved as 'classifier_performance.png'.")


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Scale data for Chi-Square and Mutual Information
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Filter Methods
def apply_filter_methods(X, y):
    results = {}

    # 1. Variance Threshold
    vt = VarianceThreshold(threshold=0.01)
    X_vt = vt.fit_transform(X)
    results['Variance Threshold'] = X_vt

    # 2. Correlation
    corr_matrix = np.corrcoef(X.T, y, rowvar=True)
    corr_threshold = 0.2
    selected_features_corr = [i for i in range(X.shape[1]) if abs(corr_matrix[i, -1]) > corr_threshold]
    X_corr = X[:, selected_features_corr]
    results['Correlation'] = X_corr

    # 3. Chi-Square
    chi2_selector = SelectKBest(score_func=chi2, k=10)
    X_chi2 = chi2_selector.fit_transform(X_scaled, y)
    results['Chi-Square'] = X_chi2

    # 4. Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_mi = mi_selector.fit_transform(X_scaled, y)
    results['Mutual Information'] = X_mi

    return results

# Classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

# Evaluate Models
def evaluate_models(X_train, X_test, y_train, y_test, classifiers):
    results = []
    for name, clf in classifiers.items():
        model = clone(clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Classifier": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
    return results

# Main Execution
filter_results = apply_filter_methods(X, y)

final_results = {}
for filter_name, X_filtered in filter_results.items():
    print(f"\nApplying Filter: {filter_name}")
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=42)

    results = evaluate_models(X_train, X_test, y_train, y_test, classifiers)
    final_results[filter_name] = results

    for result in results:
        print(f"Classifier: {result['Classifier']}, Accuracy: {result['Accuracy']:.2f}, "
              f"Precision: {result['Precision']:.2f}, Recall: {result['Recall']:.2f}, F1 Score: {result['F1 Score']:.2f}")

# Convert results into a DataFrame for each filter
all_results = []
for filter_name, results in final_results.items():
    for result in results:
        result['Filter'] = filter_name
        all_results.append(result)

df_results = pd.DataFrame(all_results)

# Function to save results as a PNG image with both table and bar plot
def save_results_to_png(df, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 2]})

    # Barplot for classifier performance metrics
    sns.set(style="whitegrid")
    sns.barplot(
        data=df.melt(id_vars=["Classifier", "Filter"], 
                     value_vars=["Accuracy", "Precision", "Recall", "F1 Score"], 
                     var_name="Metric", 
                     value_name="Score"),
        x="Metric", 
        y="Score", 
        hue="Classifier", ax=ax1
    )
    ax1.set_title("Performance of Classifiers by Metric and Filter")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.legend(loc='upper right')

    # Table for classifier performance scores
    table = ax2.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    ax2.axis("off")  # Hide axes for the table

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

save_results_to_png(df_results, "filter_classifier_performance_with_table.png")
print("Results saved as 'filter_classifier_performance_with_table.png'.")
