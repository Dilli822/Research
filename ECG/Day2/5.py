import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt

# Bandpass filter function (0.5-50 Hz)
def bandpass_filter(signal, fs, lowcut=0.5, highcut=50.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Normalize the ECG signal to [0, 1] range
def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# List of multiple ECG record paths
record_paths = [
    'ecg-id-database-1.0.0/Person_08/rec_1',
    'ecg-id-database-1.0.0/Person_08/rec_2'
]

# Dictionary to store spectrograms for each record
spectrogram_dict = {}

for record_path in record_paths:
    # Load ECG record
    record = wfdb.rdrecord(record_path)
    
    # Extract raw signal (first channel)
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs  # Sampling frequency
    
    # Plot raw ECG signal
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_signal, color='navy')
    plt.title(f'Raw ECG Signal - {record_path}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Bandpass filter the ECG signal (remove noise)
    filtered_signal = bandpass_filter(ecg_signal, fs)
    
    # Normalize the filtered signal
    normalized_signal = normalize_signal(filtered_signal)
    
    # Plot the filtered and normalized ECG signal
    plt.figure(figsize=(12, 4))
    plt.plot(normalized_signal, color='darkgreen')
    plt.title(f'Filtered and Normalized ECG Signal - {record_path}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Generate spectrogram
    frequencies, times, Sxx = spectrogram(normalized_signal, fs)
    spectrogram_2d = 10 * np.log10(Sxx + 1e-10)  # Convert to decibels

    # Normalize the spectrogram to [0, 1]
    spectrogram_2d_normalized = (spectrogram_2d - np.min(spectrogram_2d)) / (np.max(spectrogram_2d) - np.min(spectrogram_2d))

    # Store the spectrogram in the dictionary with the record_path as the key
    spectrogram_dict[record_path] = spectrogram_2d_normalized

    print(f"\n Normalized Spectrogram stored for {record_path}:")
    print("Spectrogram shape:", spectrogram_2d_normalized.shape)
    print("Spectrogram dimensions:", spectrogram_2d_normalized.ndim)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, spectrogram_2d_normalized, shading='gouraud' ,cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Normalized Spectrogram of ECG Signal for {record_path} (2D)')
    plt.colorbar(label='NormalizedPower [dB]')
    plt.tight_layout()
    plt.show()

# Example: Printing the stored spectrogram for a specific record
print("\nStored Spectrograms:")
for record_path, spectrogram_2d in spectrogram_dict.items():
    print(f"\nSpectrogram for {record_path}:")
    print(spectrogram_2d[:25])  # Print the 2D spectrogram array for each record
