import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

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

# Load ECG record
record_paths = [
    'ecg-id-database-1.0.0/Person_08/rec_1',
    'ecg-id-database-1.0.0/Person_08/rec_2'
]

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

    # Detect R-peaks (heartbeats)
    peaks, _ = find_peaks(normalized_signal, height=0.5, distance=fs*0.6)  # R-peak detection with threshold
    
    # Plot the ECG signal with R-peaks
    plt.figure(figsize=(12, 4))
    plt.plot(normalized_signal, color='darkgreen')
    plt.plot(peaks, normalized_signal[peaks], 'ro', label='R-peaks')
    plt.title(f'ECG Signal with R-Peaks - {record_path}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Normalized Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Segment the ECG signal into heartbeats (R-R intervals)
    heartbeats = []
    for i in range(1, len(peaks)):
        start_idx = peaks[i-1]
        end_idx = peaks[i]
        heartbeat = normalized_signal[start_idx:end_idx]
        heartbeats.append(heartbeat)

    print(f"\nNumber of heartbeats (R-R intervals) in {record_path}: {len(heartbeats)}")
    
    # Example: Plot the first heartbeat segment
    plt.figure(figsize=(10, 4))
    plt.plot(heartbeats[0], color='purple')
    plt.title(f"First Heartbeat Segment from {record_path}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

