import h5py

# File path for the .h5 file
file_path = 'ecg_results/ecg_biometric_p01_r1.h5'

# Open the .h5 file to inspect its contents
with h5py.File(file_path, 'r') as file:
    # Print all top-level keys in the HDF5 file
    print("Keys in the file:", list(file.keys()))
    
    # If you want to inspect the contents of each group/dataset:
    for key in file.keys():
        print(f"\nDataset name: {key}")
        print(f"Shape: {file[key].shape}")
        print(f"Data type: {file[key].dtype}")

