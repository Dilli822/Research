import pandas as pd

# Function to convert .data file to .csv without modifying columns
def convert_data_to_csv(input_file, output_file):
    # Load the dataset without column modifications
    data = pd.read_csv(input_file, header=None)  # Read as-is with no header
    
    # Save the dataframe to a CSV file
    data.to_csv(output_file, index=False)
    print(f"Data successfully converted and saved to {output_file}")

# Example usage
input_file = 'wpbc.data'  # Replace with your actual .data file path
output_file = 'wpbc.csv'  # Output .csv file
convert_data_to_csv(input_file, output_file)
