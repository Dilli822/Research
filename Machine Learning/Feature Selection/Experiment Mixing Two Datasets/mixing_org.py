import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Heart and Pima datasets
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Define column names for Heart and Pima datasets
heart_columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
pima_columns = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 
    'bmi', 'diabetes_pedigree', 'age', 'outcome'
]

# Load datasets with specified column names
heart_data = pd.read_csv(heart_url, names=heart_columns, na_values='?')
pima_data = pd.read_csv(pima_url, names=pima_columns)

# Drop rows with missing values in Heart dataset
heart_data = heart_data.dropna()

# Preprocess: Remove target columns and standardize both datasets
heart_features = heart_data.drop(columns=['target'])
pima_features = pima_data.drop(columns=['outcome'])
heart_target = heart_data['target']

# Standardize datasets
heart_features = (heart_features - heart_features.mean()) / heart_features.std()
pima_features = (pima_features - pima_features.mean()) / pima_features.std()

# Rename columns to ensure uniqueness
heart_features = heart_features.add_prefix("heart_")
pima_features = pima_features.add_prefix("pima_")

# Combine selected columns for the new mixed dataset
# Ensure the number of rows is equal by sampling the same number from both datasets
min_rows = min(heart_features.shape[0], pima_features.shape[0])

# Randomly sample the same number of rows from both datasets
heart_selected_rows = heart_features.sample(n=min_rows, random_state=42)
pima_selected_rows = pima_features.sample(n=min_rows, random_state=42)

# Combine the selected rows into a new mixed dataset
new_mixed_dataset = pd.concat([heart_selected_rows.reset_index(drop=True), 
                               pima_selected_rows.reset_index(drop=True), 
                               heart_target.sample(n=min_rows, random_state=42).reset_index(drop=True)], 
                              axis=1)

new_mixed_dataset.rename(columns={'target': 'heart_target'}, inplace=True)

# Print selected columns for verification
print("Heart_selected_cols:\n", heart_selected_rows)
print("Pima_selected_cols:\n", pima_selected_rows)

print("\nNew Mixed Dataset:\n", new_mixed_dataset)

# Save the new mixed dataset to a CSV file
output_file = "new_mixed_dataset.csv"
new_mixed_dataset.to_csv(output_file, index=False)

print(f"New mixed dataset saved to {output_file}")

# Pearson's correlation function
def pearson_correlation(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sqrt(np.sum((X - X_mean) ** 2) * np.sum((Y - Y_mean) ** 2))
    return numerator / denominator

# Calculate Pearson's r for each feature with respect to the target
selected_features = {}
threshold = 0.3
Y = new_mixed_dataset['heart_target']

for column in new_mixed_dataset.drop(columns=['heart_target']).columns:
    r_value = pearson_correlation(new_mixed_dataset[column], Y)
    selected_features[column] = {
        'correlation': r_value,
        'selected': abs(r_value) >= threshold
    }

# Display results and plot
print("Feature Correlations with Target (Heart Target):")
for feature, stats in selected_features.items():
    print(f"{feature}: Pearson's r = {stats['correlation']:.3f}, Selected: {stats['selected']}")

# Plot correlation results
plt.figure(figsize=(10, 6))
sns.barplot(x=list(selected_features.keys()), 
            y=[stats['correlation'] for stats in selected_features.values()], 
            palette="viridis")
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Selection Threshold (|r| >= {threshold})')
plt.xticks(rotation=90)
plt.ylabel("Pearson's r")
plt.title("Pearson Correlation of Each Feature with Heart Target")
plt.legend()
plt.show()

# Store all computed features with their correlation values
all_computed_features = {feature: stats['correlation'] for feature, stats in selected_features.items()}

# Separate selected and rejected based on threshold
selected_features_list = {feature: stats['correlation'] for feature, stats in selected_features.items() if stats['selected']}
rejected_features_list = {feature: stats['correlation'] for feature, stats in selected_features.items() if not stats['selected']}

# Print all selected features
print("Selected Features (|r| >= threshold):")
if selected_features_list:
    for feature, r_value in selected_features_list.items():
        print(f"{feature}: Pearson's r = {r_value:.3f}")
else:
    print("No features selected.")

# Print all rejected features
print("\nRejected Features (|r| < threshold):")
if rejected_features_list:
    for feature, r_value in rejected_features_list.items():
        print(f"{feature}: Pearson's r = {r_value:.3f}")
else:
    print("No features rejected.")
