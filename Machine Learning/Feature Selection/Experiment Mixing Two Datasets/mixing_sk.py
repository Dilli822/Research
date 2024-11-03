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

# Function to create a new mixed dataset with a specified number of columns
def create_mixed_dataset(heart_features, pima_features, heart_target, n_heart_cols, n_pima_cols):
    # Select specified number of columns from each dataset
    heart_selected_cols = heart_features.sample(n=n_heart_cols, axis=1, random_state=42)
    pima_selected_cols = pima_features.sample(n=n_pima_cols, axis=1, random_state=42)

    # Combine selected columns while ensuring equal number of rows
    min_rows = min(heart_selected_cols.shape[0], pima_selected_cols.shape[0])
    heart_selected_rows = heart_selected_cols.sample(n=min_rows, random_state=42)
    pima_selected_rows = pima_selected_cols.sample(n=min_rows, random_state=42)

    # Create new mixed dataset
    new_mixed_dataset = pd.concat([
        heart_selected_rows.reset_index(drop=True), 
        pima_selected_rows.reset_index(drop=True), 
        heart_target.sample(n=min_rows, random_state=42).reset_index(drop=True)
    ], axis=1)

    new_mixed_dataset.rename(columns={'target': 'heart_target'}, inplace=True)
    return new_mixed_dataset

# Set the number of columns to mix from each dataset
n_heart_cols = 1  # Change this value as needed
n_pima_cols = 5   # Change this value as needed

# Create the new mixed dataset
new_mixed_dataset = create_mixed_dataset(heart_features, pima_features, heart_target, n_heart_cols, n_pima_cols)

# Save the new mixed dataset to a CSV file
output_file = "new_mixed_dataset.csv"
new_mixed_dataset.to_csv(output_file, index=False)
print(f"Mixed dataset saved to {output_file}.")

# Function to calculate Fisher's score
def fisher_score(X, y):
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    
    for i in range(n_features):
        feature_values = X.iloc[:, i]

        # Calculate means and variances for the overall feature
        mean_total = np.mean(feature_values)
        variance_total = np.var(feature_values)

        # Initialize lists to hold mean and variance per class
        mean_per_class = []
        variance_per_class = []
        n_per_class = []
        
        for label in class_labels:
            class_data = feature_values[y == label]
            mean_per_class.append(np.mean(class_data))
            variance_per_class.append(np.var(class_data))
            n_per_class.append(len(class_data))

        # Calculate Fisher score
        numerator = sum([n * (mean - mean_total) ** 2 for n, mean in zip(n_per_class, mean_per_class)])
        denominator = sum([n * variance for n, variance in zip(n_per_class, variance_per_class)])

        if denominator > 0:
            scores[i] = numerator / denominator
        else:
            scores[i] = 0
            
    return scores

# Calculate Fisher's score for each feature
X = new_mixed_dataset.drop(columns=['heart_target'])
y = new_mixed_dataset['heart_target']
fisher_scores = fisher_score(X, y)

# Create a DataFrame to display features and their Fisher scores
fisher_scores_df = pd.DataFrame({'Feature': X.columns, 'Fisher Score': fisher_scores})

# Sort the features by their Fisher scores
fisher_scores_df = fisher_scores_df.sort_values(by='Fisher Score', ascending=False)

# Print Fisher scores
print("Fisher's Scores for Features:")
print(fisher_scores_df)

# Plot Fisher scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Fisher Score', y='Feature', data=fisher_scores_df, palette="viridis")
plt.title("Fisher's Score for Each Feature")
plt.xlabel("Fisher Score")
plt.ylabel("Feature")
plt.show()

# Select features based on a threshold for Fisher scores
threshold = 1.0  # Set your own threshold
selected_features_fisher = fisher_scores_df[fisher_scores_df['Fisher Score'] >= threshold]

# Print selected features based on Fisher's score
print("\nSelected Features based on Fisher's Score:")
if not selected_features_fisher.empty:
    for feature, score in zip(selected_features_fisher['Feature'], selected_features_fisher['Fisher Score']):
        print(f"{feature}: Fisher Score = {score:.3f}")
else:
    print("No features selected based on Fisher's score.")
