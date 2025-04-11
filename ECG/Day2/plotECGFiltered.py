import h5py
import matplotlib.pyplot as plt
import numpy as np

person_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 21, 22, 22, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 30, 30, 30, 30, 30, 31, 31, 32, 32, 32, 32, 32, 32, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 40, 40, 41, 41, 42, 42, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 46, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 57, 58, 58, 59, 59, 59, 59, 59, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 63, 63, 63, 63, 63, 63, 64, 64, 64, 65, 65, 66, 66, 67, 67, 67, 68, 68, 69, 69, 70, 70, 70, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 85]
record_nums = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 3, 4, 5, 6, 7, 8, 9, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3]

# Loop through all person_ids and record_nums
for person_id, record_num in zip(person_ids, record_nums):
    # Define the paths for the filtered and raw ECG files
    filtered_filename = f"filterdECG2/ecg_biometric_p{person_id:02d}_record{record_num}.h5"
    raw_filename = f"ecg-id-database-1.0.0/Person_{person_id:02d}/rec_{record_num}.dat"
    
    # Load the filtered ECG from the filtered file
    with h5py.File(filtered_filename, 'r') as f:
        filtered_ecg = f['filtered_ecg'][:]
    
    # Load the raw ECG signal from the .dat file
    # Assuming the ECG data in the .dat file is in a binary format and stored as float32 (you may need to adjust the dtype)
    raw_ecg = np.fromfile(raw_filename, dtype=np.float32)
    
    # Plot raw ECG in a separate figure
    plt.figure(figsize=(12, 6))
    plt.plot(raw_ecg)
    plt.title(f"Raw ECG for Person {person_id}, Record {record_num}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Dynamically adjust figure size based on data length
    plt.gcf().set_size_inches(len(raw_ecg) / 1000, 6)  # Adjust width based on data length

    plt.tight_layout()
    plt.show()

    # Plot filtered ECG in a separate figure
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_ecg)
    plt.title(f"Filtered ECG for Person {person_id}, Record {record_num}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Plotted raw and filtered ECG for Person {person_id}, Record {record_num}")
