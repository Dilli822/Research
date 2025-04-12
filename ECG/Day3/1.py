import numpy as np
import matplotlib.pyplot as plt

# File path to the LONG TERM ECG data file
# file_path = 'CYBHi/data/long-term/20120106-AA-A0-35.txt'

# File path to the SHORT TERM ECG data file
file_path = 'CYBHi/data/short-term/20110715-MLS-A1-8B.txt'

# Function to load ECG data from the text file
def load_ecg_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip lines that are part of the header
    data_start_index = lines.index('# EndOfHeader\n') + 1  # Data starts after this line
    
    # Initialize an empty list to store ECG signal values
    ecg_data = []
    
    # Extract ECG signal values (handle tab-separated values)
    for line in lines[data_start_index:]:
        # Split line by tab characters and convert each value to an integer
        values = line.strip().split('\t')  # Split by tab
        for value in values:
            try:
                ecg_data.append(int(value))  # Convert to integer and append
            except ValueError:
                # Skip invalid lines or values that cannot be converted to integer
                continue
    
    return np.array(ecg_data)

# Load the ECG data
ecg_signal = load_ecg_data(file_path)

# Inspect the first few samples of the ECG signal
samplenum = 10
print(ecg_signal[:samplenum])  # Show the first 10 samples


# Plot the first samplenum samples of the ECG signal
plt.figure(figsize=(10, 5))
plt.plot(ecg_signal[:100])
plt.title("CYBHi ECG Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
