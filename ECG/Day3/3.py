import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg

# load raw ECG signal
signal = np.loadtxt('CYBHi/data/long-term/20120106-AL-A0-35.txt')

# 2. Use BioSPPy's ECG function to filter and process the signal The bandpass filter (0.5–40 Hz) is applied 
# automatically by the BioSPPy library's ecg() function. 
out = ecg.ecg(signal=signal, sampling_rate=1000., show=False)

# The filtered signal is in the 'filtered' key of the output dictionary out['filtered']: 
# This extracts the filtered ECG signal (which has undergone the 0.5–40 Hz bandpass filter) 
# and stores it in the variable filtered_ecg.
filtered_ecg = out['filtered']

# 3. Segment the filtered ECG signal into 5-second windows and normalize using numpy's built-in functions
# Segmenting the signal into 5-second windows and normalizing to [0, 1] is done in the custom segment_and_normalize() function:
def segment_and_normalize(ecg_filtered, fs=1000, window_sec=5):
    win_len = fs * window_sec
    segments = []
    for start in range(0, len(ecg_filtered), win_len):
        end = start + win_len
        if end <= len(ecg_filtered):
            segment = ecg_filtered[start:end]
            # Normalize to [0, 1] using inbuilt numpy functions
            norm_segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-8)
            segments.append(norm_segment)
    return segments

# 4. Segment and normalize the signal
segments = segment_and_normalize(filtered_ecg, fs=1000, window_sec=5)
print(f"Number of segments: {len(segments)}")
print(f" segments:", segments[:4])
c_segments = np.array(segments)  # Convert to NumPy array
print(f"segments dimension is : {c_segments.ndim}") #now this will work.

# Print total number of segments
print(f"Total 5-second segments: {len(segments)}")

# Optional: Plot one segment for visualization
plt.plot(segments[0])
plt.title('Example of a 5-second normalized ECG segment')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()
