import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

# Load raw ECG signal (replace with synthetic data if file unavailable)
try:
    ecg_signal = np.loadtxt('CYBHi/data/long-term/20120106-AL-A0-35.txt')
except FileNotFoundError:
    # Synthetic ECG-like signal for testing (60 seconds)
    t = np.arange(0, 60, 1/1000)
    ecg_signal = 0.5 * np.sin(2 * np.pi * 1 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
    print("Using synthetic signal due to missing file.")

# 1. Add Gaussian noise for SNR = 20 dB
def add_gaussian_noise(signal, snr_db=20):
    # Calculate signal power (mean square)
    signal_power = np.mean(signal ** 2)
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    # Calculate noise power
    noise_power = signal_power / snr_linear
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    # Add noise to signal
    noisy_signal = signal + noise
    return noisy_signal

noisy_signal = add_gaussian_noise(ecg_signal, snr_db=20)

# 2. Bandpass filter (0.5–40 Hz)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def apply_bandpass(signal, lowcut, highcut, fs):
    sos = butter_bandpass(lowcut, highcut, fs)
    filtered = sosfilt(sos, signal)
    return filtered

fs = 1000  # Sampling rate in Hz
filtered_ecg = apply_bandpass(noisy_signal, lowcut=0.5, highcut=40.0, fs=fs)

# 3. Segment and normalize into 5-second windows
def segment_and_normalize(ecg_filtered, fs=1000, window_sec=5):
    win_len = fs * window_sec
    segments = []
    for start in range(0, len(ecg_filtered), win_len):
        end = start + win_len
        if end <= len(ecg_filtered):
            segment = ecg_filtered[start:end]
            # Normalize to [0, 1]
            norm_segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-8)
            segments.append(norm_segment)
    return segments

segments = segment_and_normalize(filtered_ecg, fs=fs, window_sec=5)

# 4. Print results
print(f"Number of segments: {len(segments)}")
print(f"First 4 segments (shape of each): {[seg.shape for seg in segments[:4]]}")
c_segments = np.array(segments)
print(f"Segments dimension: {c_segments.ndim}")

# 5. Plot one segment and a portion of noisy vs. filtered signal
plt.figure(figsize=(12, 8))

# Plot noisy vs. filtered signal (first 5 seconds)
time = np.arange(0, 5, 1/fs)
plt.subplot(2, 1, 1)
plt.plot(time, noisy_signal[:5000], label='Noisy Signal (SNR 20 dB)')
plt.plot(time, filtered_ecg[:5000], label='Filtered Signal (0.5–40 Hz)')
plt.title('Noisy vs. Filtered ECG Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()

# Plot normalized segment
plt.subplot(2, 1, 2)
plt.plot(time, segments[0])
plt.title('5-second Normalized ECG Segment (Post-Filtering)')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Amplitude')
plt.tight_layout()
plt.show()
