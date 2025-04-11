

"""
Preprocessing:
Noise Reduction:

Apply filtering techniques (e.g., bandpass filters or wavelet denoising) to remove noise and artifacts.

Address common issues like baseline wander, power line interference, and muscle artifacts.

Segmentation:

Detect individual heartbeats (QRS complexes) using algorithms like the Pan-Tompkins algorithm.

Normalization:

Normalize ECG signals to a consistent amplitude range.

Resample the signals to a uniform sampling rate.



"""


import wfdb
import numpy as np
from scipy import signal
import pywt
import neurokit2 as nk  # Pan-Tompkins algorithm
import matplotlib.pyplot as plt

# Load WFDB record
record = wfdb.rdrecord('ecg-id-database-1.0.0/Person_08/rec_1')
raw_ecg = record.p_signal[:, 0]  # First channel
fs = int(record.fs)              # Sampling frequency

# --- 1. Noise Reduction ---
def denoise_ecg(raw_signal, fs):
    # Bandpass filter to remove baseline wander, power line, muscle noise
    b, a = signal.butter(4, [0.5/(fs/2), 40/(fs/2)], btype='bandpass')
    filtered = signal.filtfilt(b, a, raw_signal)

    # Wavelet denoising
    coeffs = pywt.wavedec(filtered, 'db4', level=5)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(filtered)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, 'db4')

# --- 2. R-peak Segmentation ---
def detect_r_peaks(cleaned_signal, fs):
    _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=fs)
    return rpeaks["ECG_R_Peaks"]

# --- 3. Heartbeat Extraction ---
def extract_beats(signal, r_peaks, fs):
    win_size = int(0.6 * fs)
    half = win_size // 2
    beats = []
    for r in r_peaks:
        if r-half > 0 and r+half < len(signal):
            beats.append(signal[r-half:r+half])
    return np.array(beats)

# --- 4. Normalization ---
def normalize_beats(beats):
    beats = (beats - np.mean(beats, axis=1, keepdims=True)) / np.std(beats, axis=1, keepdims=True)
    target_len = 150  # Resample each beat to same length
    return np.array([signal.resample(b, target_len) for b in beats])

# --- Full Preprocessing ---
def preprocess_ecg(raw_ecg, fs):
    cleaned = denoise_ecg(raw_ecg, fs)
    r_peaks = detect_r_peaks(cleaned, fs)
    beats = extract_beats(cleaned, r_peaks, fs)
    norm_beats = normalize_beats(beats)
    return norm_beats, r_peaks, cleaned

# Run the pipeline
beats, r_peaks, cleaned = preprocess_ecg(raw_ecg, fs)

# --- Print Details ---
print("\n--- Raw ECG Sample ---")
print(raw_ecg[:10])  # Print first 10 samples
print(f"Length of raw ECG: {len(raw_ecg)} samples")

print("\n--- Denoised ECG (first 10 samples) ---")
print(cleaned[:10])  # Print first 10 samples of denoised ECG

print("\n--- Detected R-Peak Indices ---")
print(r_peaks[:10])  # Print first 10 detected R-peaks
print(f"Number of detected R-peaks: {len(r_peaks)}")

print("\n--- Number of Extracted Beats ---")
print(len(beats))  # Number of heartbeats extracted

print("\n--- Shape of One Beat (after normalization & resampling) ---")
print(beats[0].shape if len(beats) > 0 else "No beats extracted")

print("\n--- First Normalized Beat (first 10 samples) ---")
print(beats[0][:10] if len(beats) > 0 else "No beat data")

print("\n--- ECG Preprocessing Complete ---")


# --- Visualization ---
# Raw vs Cleaned
plt.figure(figsize=(12, 4))
plt.plot(raw_ecg[:1000], label='Raw ECG', alpha=0.5)
plt.plot(cleaned[:1000], label='Denoised ECG', color='orange')
plt.legend(); plt.title('Raw vs Denoised ECG')
plt.xlabel("Sample"); plt.ylabel("Amplitude"); plt.grid(True)
plt.tight_layout(); plt.show()

# R-peak detection
plt.figure(figsize=(12, 4))
plt.plot(cleaned, label='Denoised ECG')
plt.plot(r_peaks, cleaned[r_peaks], 'ro', label='R-peaks')
plt.legend(); plt.title('R-Peak Detection')
plt.xlabel("Sample"); plt.ylabel("Amplitude"); plt.grid(True)
plt.tight_layout(); plt.show()

# Show 5 normalized heartbeats
plt.figure(figsize=(12, 4))
for i in range(min(5, len(beats))):
    plt.plot(beats[i], label=f'Beat {i+1}')
plt.legend(); plt.title('First 5 Normalized Heartbeats')
plt.xlabel("Sample"); plt.ylabel("Normalized Amplitude")
plt.tight_layout(); plt.show()


# Calculate the time duration of the raw ECG signal
num_samples = len(raw_ecg)
time_duration = num_samples / fs

# Print the time duration in seconds
print(f"Time duration of the ECG signal: {time_duration:.2f} seconds")



# Assuming 'r_peaks' are detected and 'fs' is the sampling frequency
# To extract PQRS waves, let's assume we extract the window from 100ms before R-peak to 400ms after R-peak.

# Define the window for each wave component (in samples)
p_wave_duration = int(0.12 * fs)  # 120 ms before the R-peak for P-wave
qrs_duration = int(0.12 * fs)     # 120 ms around the R-peak for QRS complex
t_wave_duration = int(0.2 * fs)   # 200 ms after the R-peak for T-wave

# Create a plot showing the PQRS waves for the first detected R-peak
plt.figure(figsize=(12, 6))

for i, r_peak in enumerate(r_peaks[:5]):  # Loop through first 5 R-peaks (you can change this)
    # P-wave: 100 ms before the R-peak
    p_wave = raw_ecg[r_peak - p_wave_duration: r_peak]
    
    # QRS complex: around the R-peak (centered around R-peak)
    qrs_wave = raw_ecg[r_peak - qrs_duration: r_peak + qrs_duration]
    
    # T-wave: 200 ms after the R-peak
    t_wave = raw_ecg[r_peak + qrs_duration: r_peak + qrs_duration + t_wave_duration]

    # Plot the P, QRS, and T waves
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(p_wave)) / fs, p_wave, label=f'P-wave {i+1}')
    plt.title('P-wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(qrs_wave)) / fs, qrs_wave, label=f'QRS Complex {i+1}')
    plt.title('QRS Complex')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(t_wave)) / fs, t_wave, label=f'T-wave {i+1}')
    plt.title('T-wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.show()



# Define the window size for each wave (in samples)
# For 1 second, with the sampling frequency (fs)
one_second_samples = fs  # 1 second = fs samples

# Define the offset durations
p_wave_duration = int(0.5 * fs)  # 500 ms before the R-peak for P-wave
qrs_duration = one_second_samples  # 1 second around the R-peak for QRS complex
t_wave_duration = int(0.5 * fs)   # 500 ms after the R-peak for T-wave

# Create a plot showing the PQRS waves for the first detected R-peak
plt.figure(figsize=(12, 6))

for i, r_peak in enumerate(r_peaks[:5]):  # Loop through first 5 R-peaks (you can change this)
    # P-wave: 500 ms before the R-peak
    p_wave = raw_ecg[r_peak - p_wave_duration: r_peak]
    
    # QRS complex: 1 second around the R-peak (centered around R-peak)
    qrs_wave = raw_ecg[r_peak - p_wave_duration: r_peak + p_wave_duration]
    
    # T-wave: 500 ms after the R-peak
    t_wave = raw_ecg[r_peak + p_wave_duration: r_peak + one_second_samples]

    # Plot the P, QRS, and T waves
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(p_wave)) / fs, p_wave, label=f'P-wave {i+1}')
    plt.title(f'P-wave - Heartbeat {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(qrs_wave)) / fs, qrs_wave, label=f'QRS Complex {i+1}')
    plt.title(f'QRS Complex - Heartbeat {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(t_wave)) / fs, t_wave, label=f'T-wave {i+1}')
    plt.title(f'T-wave - Heartbeat {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.show()




# Assuming fs is the sampling frequency (e.g., 500 Hz)
# The number of samples corresponding to 1 second
one_second_samples = fs

# Create a plot showing the 1-second heartbeat centered around the first 5 R-peaks
plt.figure(figsize=(12, 6))

for i, r_peak in enumerate(r_peaks[:2]):  # Loop through first 5 R-peaks (change as needed)
    # Extract 500 samples (1 second) around the R-peak
    heartbeat = raw_ecg[r_peak - (one_second_samples // 2): r_peak + (one_second_samples // 2)]

    # Plot the full heartbeat
    plt.plot(np.arange(len(heartbeat)) / fs, heartbeat, label=f'Heartbeat {i+1}')
    plt.title(f'1-Second Heartbeat {i+1} Centered around R-peak')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.legend()
plt.show()



# Calculate R-R intervals in samples
rr_intervals_samples = np.diff(r_peaks)

# Convert R-R intervals to seconds
rr_intervals_seconds = rr_intervals_samples / fs

# Average R-R interval (cardiac cycle duration)
avg_rr_interval = np.mean(rr_intervals_seconds)
print(f"Average Cardiac Cycle Duration: {avg_rr_interval} seconds")



# Assuming fs is the sampling frequency (e.g., 500 Hz)
# The number of samples corresponding to 8 milliseconds (0.008 seconds)
samples_for_8ms = int(0.008 * fs)

# Create a plot showing the 8ms heartbeat centered around the first 2 R-peaks
plt.figure(figsize=(12, 6))

for i, r_peak in enumerate(r_peaks[:2]):  # Loop through first 2 R-peaks (change as needed)
    # Extract 4 samples (8ms) around the R-peak
    heartbeat = raw_ecg[r_peak - (samples_for_8ms // 2): r_peak + (samples_for_8ms // 2)]

    # Plot the full heartbeat
    plt.plot(np.arange(len(heartbeat)) / fs, heartbeat, label=f'Heartbeat {i+1}')
    plt.title(f'8-Millisecond Heartbeat {i+1} Centered around R-peak')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.legend()
plt.show()