import pandas as pd

# Function to count the number of feature columns (excluding the target column)
def count_feature_columns(file):
    # Load the CSV file
    df = pd.read_csv(file)
    
    # Count the number of columns excluding the target column (assuming target column is named 'diagnosis')
    num_features = len(df.columns) - 1  # Exclude the target column
    return num_features

# Function to compare two .csv files and print conflicting columns
def compare_csv_columns(file1, file2):
    # Load the two CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Count the number of feature columns in both files
    num_features_file1 = count_feature_columns(file1)
    num_features_file2 = count_feature_columns(file2)
    
    print(f"Number of feature columns in {file1}: {num_features_file1}")
    print(f"Number of feature columns in {file2}: {num_features_file2}")

    # Check if columns match
    common_columns = set(df1.columns).intersection(df2.columns)

    # Iterate through the common columns and check for conflicts
    conflicting_columns = []
    for column in common_columns:
        if not df1[column].equals(df2[column]):
            conflicting_columns.append(column)

    if conflicting_columns:
        print("Conflicting columns:")
        for col in conflicting_columns:
            print(f"Column: {col}")
    else:
        print("No conflicts found between the columns.")

# Example usage
file1 = 'wpbc.csv'  # Replace with the first CSV file path
file2 = 'wdbc.csv'  # Replace with the second CSV file path

compare_csv_columns(file1, file2)
