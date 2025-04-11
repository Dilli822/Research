import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Sampling frequency (Hz)
fs = 500

# Generate a dummy 1D ECG signal
ecg_signal = np.random.randn(1000)  # Replace with your real ECG data

# Store the 1D signal
original_signal_1d = ecg_signal

# Print details about the 1D signal
print("Original ECG signal (1D):")
print(original_signal_1d[:25])
print("Shape:", original_signal_1d.shape)
print("Dimensions:", original_signal_1d.ndim)

# Plot the original ECG signal
plt.figure(figsize=(10, 4))
plt.plot(original_signal_1d)
plt.title("Original ECG Signal (1D)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Generate spectrogram
frequencies, times, Sxx = spectrogram(ecg_signal, fs)
spectrogram_2d = 10 * np.log10(Sxx + 1e-10)  # Convert to dB, add small value to avoid log(0)

# Print details about the 2D spectrogram
print("\nConverted Spectrogram (2D):")
print("Shape:", spectrogram_2d.shape)
print("Dimensions:", spectrogram_2d.ndim)

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, spectrogram_2d, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('ECG Signal Spectrogram (2D)')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()

# Now both `original_signal_1d` and `spectrogram_2d` are stored and ready for further processing.

