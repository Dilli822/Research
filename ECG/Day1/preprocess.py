import wfdb
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import h5py


def load_ecg_record(person_id, record_num):
    """Load ECG signal and annotations from WFDB files."""
    record_path = f'ecg-id-database-1.0.0/Person_{person_id:02d}/rec_{record_num}'
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    return record, annotations

def preprocess_ecg(ecg_signal, original_fs, target_fs=250):
    """Bandpass filter (0.5-40Hz) and resample to target_fs."""
    # Resample if needed
    if original_fs != target_fs:
        ecg_signal = signal.resample(ecg_signal, int(len(ecg_signal)*target_fs/original_fs))
    
    # Bandpass filter (removes baseline wander & high-frequency noise)
    b, a = signal.butter(4, [0.5/(target_fs/2), 40/(target_fs/2)], btype='bandpass')
    filtered_ecg = signal.filtfilt(b, a, ecg_signal)
    
    return filtered_ecg

def segment_heartbeats(ecg_signal, annotations, samples_before=180, samples_after=220):
    """Extract individual heartbeats using R-peak annotations."""
    r_locations = annotations.sample
    beats = []
    
    for r in r_locations:
        if r-samples_before > 0 and r+samples_after < len(ecg_signal):
            beat = ecg_signal[r-samples_before : r+samples_after]
            beats.append(beat)
    
    return np.array(beats)

# Process Person_01, rec_1
person_id = 1
record_num = 1
record, annotations = load_ecg_record(person_id, record_num)
ecg_signal = record.p_signal[:, 0]  # Lead I
fs = record.fs

# Preprocessing
clean_ecg = preprocess_ecg(ecg_signal, fs)
beats = segment_heartbeats(clean_ecg, annotations)
template = np.mean(beats, axis=0)

# --------------------------------------------------------------------------
# FIGURE 1: Preprocessing Comparison
# --------------------------------------------------------------------------
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_signal, label='Raw ECG', color='blue', alpha=0.7)
plt.scatter(annotations.sample, ecg_signal[annotations.sample], color='red', label='R-peaks')
plt.title(f"Person {person_id}, Record {record_num} - Raw ECG (Before Preprocessing)")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(clean_ecg, label='Filtered ECG', color='green')
plt.scatter(annotations.sample, clean_ecg[annotations.sample], color='red', label='R-peaks')
plt.title(f"Person {person_id}, Record {record_num} - Processed ECG (After Preprocessing)")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# FIGURE 2: All Segmented Heartbeats (Overlay)
# --------------------------------------------------------------------------
plt.figure(figsize=(15, 6))
for i, beat in enumerate(beats):
    plt.plot(beat, alpha=0.4, label=f'Beat {i+1}' if i < 5 else None)
plt.plot(template, color='black', linewidth=3, label='Average Template')
plt.title(f"Person {person_id}, Record {record_num} - All Segmented Heartbeats (n={len(beats)})")
plt.xlabel("Samples from R-peak")
plt.ylabel("Amplitude (mV)")
plt.legend(ncol=5)
plt.grid()
plt.show()

# --------------------------------------------------------------------------
# FIGURE 3: Aligned Heartbeats (Stacked)
# --------------------------------------------------------------------------
plt.figure(figsize=(15, 6))
for beat in beats:
    plt.plot(beat, alpha=0.3, color='blue')
plt.plot(template, color='red', linewidth=3, label='Average Template')
plt.title(f"Person {person_id}, Record {record_num} - Aligned Heartbeats with Template")
plt.xlabel("Samples from R-peak")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid()
plt.show()

# --------------------------------------------------------------------------
# FIGURE 4: Template Visualization
# --------------------------------------------------------------------------
plt.figure(figsize=(15, 4))
plt.plot(template, color='red', linewidth=2, label='Average Template')
plt.title(f"Person {person_id}, Record {record_num} - Final ECG Template")
plt.xlabel("Samples from R-peak")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.legend()
plt.show()


plt.savefig('preprocessing_comparison.png', dpi=300, bbox_inches='tight')


# Step 1: Define the filename
hdf5_filename = f'ecg_biometric_p{person_id:02d}_r{record_num}.h5'

# Step 2: Create the file and write data
with h5py.File(hdf5_filename, 'w') as f:
    # Create datasets for ECG signals
    f.create_dataset('clean_ecg', data=clean_ecg)
    f.create_dataset('beats', data=beats)
    f.create_dataset('template', data=template)
    f.create_dataset('r_peaks', data=annotations.sample)

    # Write attributes (metadata)
    f.attrs['person_id'] = person_id
    f.attrs['record_num'] = record_num
    f.attrs['sampling_rate'] = fs
    f.attrs['num_beats'] = beats.shape[0]
    f.attrs['samples_per_beat'] = beats.shape[1]

