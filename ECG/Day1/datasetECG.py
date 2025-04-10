import wfdb
import os

# 🔹 1. Dataset Download & Exploration (1–2 hrs)
record = wfdb.rdrecord('ecg-id-database-1.0.0/Person_08/rec_2')  # reads .hea and .dat
annotation = wfdb.rdann('ecg-id-database-1.0.0/Person_08/rec_2', 'atr')  # reads .atr

print("Signal shape:", record.p_signal.shape)
print("Sampling frequency:", record.fs)
print("Signal names:", record.sig_name)
print("Units:", record.units)
print("Signal length:", record.sig_len)

wfdb.plot_wfdb(record=record, title='ECG Signal')

# folder_path = 'ecg-id-database-1.0.0/Person_08'
# files = os.listdir(folder_path)
# print(files)

# Extension	Full Form	Description
# .hea	Header File	Contains metadata like number of signals, sampling rate, duration, signal names, units, etc.
# .dat	Data File	Contains the actual ECG waveform data (signal values). Usually in binary format.
# .atr	Annotation File	Contains annotations, like R-peaks, rhythm types, beat labels (optional, used for training/evaluation).

# 📂 Example: For rec_1
# You need all three files together for a full ECG record:
# rec_1.hea   ➤ metadata
# rec_1.dat   ➤ signal values
# rec_1.atr   ➤ optional annotations (like beat positions)
# 🔧 Usage in Code:
# record = wfdb.rdrecord('path/to/rec_1')  # reads .hea and .dat
# annotation = wfdb.rdann('path/to/rec_1', 'atr')  # reads .atr

# ✅ Signal Details
print("Signal shape:", record.p_signal.shape)  # (samples, channels)
print("Sampling frequency:", record.fs)        # in Hz
print("Signal names:", record.sig_name)        # channel names (e.g., ['ECG'])
print("Units:", record.units)                  # units (e.g., ['mV'])
print("Signal length (samples):", record.sig_len)

# ✅ Annotation Details
print("Number of annotations:", len(annotation.sample))  # Number of beats or events
print("Annotation symbol examples:", annotation.symbol[:10])  # Beat types (e.g., 'N' for normal)
print("Annotation sample indices:", annotation.sample[:10])   # Beat positions in samples

# 🔹 3. Plot ECG with annotations (e.g., R-peaks)
wfdb.plot_wfdb(record=record, annotation=annotation, title='ECG with Annotations')

# 🔹 4. Convert sample indices to time (in seconds)
fs = record.fs
r_peak_times = np.array(annotation.sample) / fs  # Time of R-peaks

# 🔹 5. Calculate RR intervals (time between beats)
rr_intervals = np.diff(r_peak_times)  # in seconds
heart_rates = 60 / rr_intervals       # in beats per minute (BPM)

# 🔹 6. Plot heart rate over time
plt.figure(figsize=(10, 4))
plt.plot(r_peak_times[1:], heart_rates, marker='o', linestyle='-', color='green')
plt.title('Heart Rate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (BPM)')
plt.grid(True)
plt.tight_layout()
plt.show()
