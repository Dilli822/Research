import wfdb
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

# Load ECG record using the wfdb library
def load_ecg_record(person_id, record_num):
    record_path = f'ecg-id-database-1.0.0/Person_{person_id:02d}/rec_{record_num}'
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    return record, annotations

# Preprocess ECG signal with resampling and bandpass filtering
def preprocess_ecg(ecg_signal, original_fs, target_fs=250):
    if original_fs != target_fs:
        ecg_signal = signal.resample(ecg_signal, int(len(ecg_signal) * target_fs / original_fs))
    b, a = signal.butter(4, [0.5 / (target_fs / 2), 40 / (target_fs / 2)], btype='bandpass')
    return signal.filtfilt(b, a, ecg_signal)

# Segment the ECG signal around the R-peaks based on annotations
def segment_heartbeats(ecg_signal, annotations, samples_before=180, samples_after=220):
    r_locations = annotations.sample
    beats = [ecg_signal[r - samples_before: r + samples_after] 
             for r in r_locations 
             if r - samples_before > 0 and r + samples_after < len(ecg_signal)]
    return np.array(beats)

# Time-domain features extraction
def extract_time_domain_features(heartbeat):
    mean_val = np.mean(heartbeat)
    std_val = np.std(heartbeat)
    skew_val = skew(heartbeat)
    kurt_val = kurtosis(heartbeat)
    return [mean_val, std_val, skew_val, kurt_val]

# Frequency-domain features extraction
def extract_frequency_domain_features(heartbeat, fs=250):
    f, Pxx = signal.welch(heartbeat, fs, nperseg=256)
    total_power = np.sum(Pxx)
    peak_frequency = f[np.argmax(Pxx)]
    return [total_power, peak_frequency]

# RR interval calculation
def extract_rr_intervals(annotations, heartbeat_length, fs=250):
    # Calculate RR intervals (difference between successive R-peak locations in time)
    r_locations = annotations.sample
    rr_intervals = np.diff(r_locations) / fs  # in seconds
    return np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

# Specify the persons and record numbers to process
person_ids = [11]
record_nums = [1]

# Process each person's ECG record
features = []
for person_id in person_ids:
    for record_num in record_nums:
        # Load ECG record and annotations
        record, annotations = load_ecg_record(person_id, record_num)
        
        # Extract ECG signal and original sampling frequency
        ecg_signal = record.p_signal[:, 0]  # Assuming ECG is in the first channel
        original_fs = record.fs
        
        # Preprocess the ECG signal
        filtered_ecg = preprocess_ecg(ecg_signal, original_fs)
        
        # Segment heartbeats from the filtered ECG
        heartbeats = segment_heartbeats(filtered_ecg, annotations)
        
        # Feature engineering for each heartbeat
        for heartbeat in heartbeats:
            time_features = extract_time_domain_features(heartbeat)
            freq_features = extract_frequency_domain_features(heartbeat)
            rr_interval = extract_rr_intervals(annotations, len(heartbeat))
            
            # Combine all features into a single feature vector
            feature_vector = time_features + freq_features + [rr_interval]
            features.append(feature_vector)
        
# Convert features list into a numpy array
features = np.array(features)
print(f"Extracted feature shape: {features.shape}")
print("feature is : ", features)
# Example: to check the number of dimensions (axes)
print("Number of dimensions: ", features.ndim)

# Check if it's a vector and what dimensions it has
if len(features.shape) == 1:
    print("It is a 1D vector with length:", features.shape[0])
elif len(features.shape) == 2:
    print("It is a 2D matrix with dimensions:", features.shape)
else:
    print("It has more than 2 dimensions")
    
# Reshape features to (samples, height, width, channels) to match MobileNetV1 input
# This assumes you have 1 channel and each sample is a 7-dimensional vector
features_reshaped = features.reshape((features.shape[0], 1, 7, 1))

# Now features_reshaped can be fed into MobileNetV1
print("Reshaped features for MobileNetV1:", features_reshaped.shape)
# Example: to check the number of dimensions (axes)
print("Number of dimensions: ", features_reshaped.ndim)

# Check if it's a vector and what dimensions it has
if len(features_reshaped.shape) == 1:
    print("It is a 1D vector with length:", features_reshaped.shape[0])
elif len(features_reshaped.shape) == 2:
    print("It is a 2D matrix with dimensions:", features_reshaped.shape)
else:
    print("It has more than 2 dimensions")
    
print("features reshaped", features_reshaped)