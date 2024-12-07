import pandas as pd

# Load each dataset
def load_datasets():
    # Define column names based on dataset structure
    # For example, WDBC dataset columns are well known in the literature:
    columns_wdbc = [
        'id', 'diagnosis', 'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
        'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points',
        'mean_symmetry', 'mean_fractal_dimension', 'radius_se', 'texture_se', 'perimeter_se',
        'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
        'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    # Load datasets, specifying column names
    dataset_1 = pd.read_csv('wdbc.csv', header=None, names=columns_wdbc)
    dataset_2 = pd.read_csv('wpbc_final_with_column_names.csv', header=None, names=columns_wdbc)  # Assuming same structure as wdbc.data

    # Return a list of the datasets
    return [dataset_1, dataset_2]

# Function to clean and standardize the column names
def clean_column_names(df):
    # Make all column names lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

# Function to combine datasets
def combine_datasets(datasets):
    # Clean column names for all datasets
    cleaned_datasets = [clean_column_names(df) for df in datasets]

    # Concatenate all datasets
    combined_data = pd.concat(cleaned_datasets, ignore_index=True)

    # Optionally, handle missing values (e.g., fill with NaN or drop rows)
    combined_data = combined_data.dropna()  # Drop rows with missing values

    return combined_data

# Main program execution
if __name__ == "__main__":
    # Load datasets
    datasets = load_datasets()

    # Combine datasets
    combined_data = combine_datasets(datasets)

    # Show combined data
    print(combined_data.head())  # Display the first few rows of the combined dataset
    print(f"Combined dataset shape: {combined_data.shape}")

    # Optionally, save the combined dataset to a CSV file
    combined_data.to_csv('combined_breast_cancer_data.csv', index=False)
