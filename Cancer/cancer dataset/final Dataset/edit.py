import pandas as pd

# Load the CSV file
df = pd.read_csv('wpbc_final.csv')

# Define the column names for features
column_names = [
    'id', 'diagnosis', 
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
    'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 
    'mean_fractal_dimension', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 
    'symmetry_worst', 'fractal_dimension_worst'
]

# Assign the column names to the DataFrame
df.columns = column_names

# Optionally, check the first few rows to confirm
print(df.head())

# If you need to save the DataFrame with the new column names
df.to_csv('wpbc_final_with_column_names.csv', index=False)
