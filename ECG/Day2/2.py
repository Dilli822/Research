import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram  # Import the spectrogram function

# Load ECG record
record_path = 'ecg-id-database-1.0.0/Person_08/rec_1'
record = wfdb.rdrecord(record_path)

# Extract raw signal (first channel)
ecg_signal = record.p_signal[:, 0]

# Plot only the raw ECG signal
plt.figure(figsize=(12, 4))
plt.plot(ecg_signal, color='navy')
plt.title('Raw ECG Signal - Person 08')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Load the record
record = wfdb.rdrecord(record_path)

# Plot the ECG signal (all channels, default layout)
wfdb.plot_wfdb(record=record, title='ECG Signal - Person 08')


# Extract the signal (assume channel 0, you can check others too)
ecg_signal = record.p_signal[:, 0]  # ECG values as a 1D NumPy array
fs = record.fs  # Sampling frequency

# Store original signal
original_signal_1d = ecg_signal

print("Original ECG signal (1D):")
print("value of spectrogram 2D: \n", original_signal_1d)
print("Shape:", original_signal_1d.shape)
print("Dimensions:", original_signal_1d.ndim)

# Plot the original ECG signal
plt.figure(figsize=(10, 4))
plt.plot(original_signal_1d)
plt.title("Original ECG Signal from ECG-ID Dataset")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Generate spectrogram
frequencies, times, Sxx = spectrogram(ecg_signal, fs)
spectrogram_2d = 10 * np.log10(Sxx + 1e-10)  # Convert to decibels

print("\nConverted Spectrogram (2D):")
print("value of spectrogram 2D: \n", spectrogram_2d)
print("Shape:", spectrogram_2d.shape)
print("Dimensions:", spectrogram_2d.ndim)

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, spectrogram_2d, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of ECG Signal (2D)')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()
