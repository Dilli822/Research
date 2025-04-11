import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# List of multiple ECG record paths
record_paths = [
    'ecg-id-database-1.0.0/Person_08/rec_1',
    'ecg-id-database-1.0.0/Person_08/rec_2',
]

for record_path in record_paths:
    # Load ECG record
    record = wfdb.rdrecord(record_path)
    
    # Extract raw signal (first channel)
    ecg_signal = record.p_signal[:, 0]

    # Plot only the raw ECG signal
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_signal, color='navy')
    plt.title(f'Raw ECG Signal - {record_path}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the ECG signal (all channels, default layout)
    wfdb.plot_wfdb(record=record, title=f'ECG Signal - {record_path}')

    # Extract the signal (assume channel 0, you can check others too)
    ecg_signal = record.p_signal[:, 0]  # ECG values as a 1D NumPy array
    fs = record.fs  # Sampling frequency

    # Store original signal
    original_signal_1d = ecg_signal

    print(f"Original ECG signal (1D) for {record_path}:")
    print("First 10 values of ECG signal: \n", original_signal_1d[:10])  # Print the first 10 values
    print("Shape:", original_signal_1d.shape)
    print("Dimensions:", original_signal_1d.ndim)

    # Plot the original ECG signal
    plt.figure(figsize=(10, 4))
    plt.plot(original_signal_1d)
    plt.title(f"Original ECG Signal from {record_path}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Generate spectrogram
    frequencies, times, Sxx = spectrogram(ecg_signal, fs)
    spectrogram_2d = 10 * np.log10(Sxx + 1e-10)  # Convert to decibels

    print(f"\nConverted Spectrogram (2D) for {record_path}:")
    print("Spectrogram shape:", spectrogram_2d.shape)
    print("Spectrogram dimensions:", spectrogram_2d.ndim)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, spectrogram_2d, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram of ECG Signal for {record_path} (2D)')
    plt.colorbar(label='Power [dB]')
    plt.tight_layout()
    plt.show()
