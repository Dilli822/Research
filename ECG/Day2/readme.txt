Loads ECG Data: Reads ECG signals and annotations from the ECG-ID Database for specified person IDs and record numbers using wfdb.

Preprocesses ECG:
Resampling: Standardizes the signal to 250 Hz if the original sampling rate differs.

Bandpass Filtering: Applies a 0.5–40 Hz filter to remove baseline wander and high-frequency noise.

Segments Heartbeats: Extracts individual heartbeats around R-peaks (180 samples before, 220 samples after) using annotation data.

Organizes Data: Creates a dictionary mapping each person ID to their list of record numbers for structured data handling.

