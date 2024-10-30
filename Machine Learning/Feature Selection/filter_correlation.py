

# import pandas as pd

# # URL of the dataset
# data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# # Read the CSV file from the URL
# df = pd.read_csv(data_url, header=None)

# # Save the DataFrame to a local CSV file
# df.to_csv("pima_indians_diabetes.csv", index=False, header=False)

# print("File downloaded and saved as 'pima_indians_diabetes.csv'")


# import pandas as pd
# import numpy as np

# # Import the CSV file
# df = pd.read_csv("pima_indians_diabetes.csv", header=None)

# # Add a new column with random floating-point values (three decimal places)
# df['random_float'] = np.round(np.random.rand(len(df)) * 1000, 2)

# # Save the modified DataFrame to a new CSV file
# df.to_csv("pima_noisy.csv", index=False)

# print("File saved as pima_noisy.csv")


# import pandas as pd
# import numpy as np

# # Import the CSV file
# df = pd.read_csv("pima_indians_diabetes.csv", header=None)

# # Generate random floating-point values (three decimal places)
# random_values = np.round(np.random.rand(len(df)) * 1000, 2)

# # Insert the new column in the second last position
# df.insert(len(df.columns) - 1, '8', random_values)

# # Save the modified DataFrame to a new CSV file
# df.to_csv("pima_noisy.csv", index=False)

# print("File saved as pima_noisy.csv")

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv("pima_noisy.csv")

# Separate features and target
X = df.iloc[:, :8]  # all features (columns 0 to 7)
y = df.iloc[:, 8]   # target (column 8)

# Use SelectKBest with a scoring function based on regression
pipeline = make_pipeline(StandardScaler(), SelectKBest(score_func=f_regression, k='all'))

# Fit the pipeline
pipeline.fit(X, y)

# Get the scores
scores = pipeline.named_steps['selectkbest'].scores_

# Create a DataFrame to visualize features and their corresponding scores
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})

# Sort by score
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# All features will be selected since we are using k='all'
selected_features = feature_scores  # All features will be selected

# Identify not selected features (none in this case since k='all')
not_selected_features = pd.DataFrame(columns=feature_scores.columns)  # Empty DataFrame since all are selected

# Display selected and not selected features
print("Selected Features:")
print(selected_features)

print("\nNot Selected Features:")
print(not_selected_features)



# Separate features and target
X = df.iloc[:, :8]  # all features (columns 0 to 7)
y = df.iloc[:, 8]   # target (column 8)

def pearson_correlation(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(x ** 2)
    sum_y2 = sum(y ** 2)
    sum_xy = sum(x * y)
    
    # Calculate the Pearson correlation coefficient using the formula
    r = (n * sum_xy - sum_x * sum_y) / ((n * sum_x2 - sum_x ** 2) ** 0.5 * (n * sum_y2 - sum_y ** 2) ** 0.5)
    
    return r

# Calculate Pearson correlation for each feature with the target variable
correlation_results = {}
for column in X.columns:
    correlation = pearson_correlation(X[column], y)
    correlation_results[column] = correlation

# Create a DataFrame to visualize features and their corresponding correlation coefficients
correlation_scores = pd.DataFrame({'Feature': X.columns, 'Correlation': correlation_results.values()})

# Sort by correlation
correlation_scores = correlation_scores.sort_values(by='Correlation', ascending=False)

# Display the results
print("Pearson Correlation Coefficients:")
print(correlation_scores)