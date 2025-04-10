import wfdb  # Library for working with PhysioNet data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest

# 🔹 1. Load the ECG signal (.hea + .dat files)
record = wfdb.rdrecord('ecg-id-database-1.0.0/Person_08/rec_2')

# 🔹 2. Load the annotations (.atr file)
annotation = wfdb.rdann('ecg-id-database-1.0.0/Person_08/rec_2', 'atr')

# Plot the ECG signal
plt.figure(figsize=(10, 4))
plt.plot(record.p_signal)
plt.title("ECG Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()

# Plot the ECG signal with annotation (R-peaks)
wfdb.plot_wfdb(record=record, annotation=annotation, title='ECG with Annotations')

# Signal Analysis for Biometric Features | Calculate RR intervals (in seconds)
rr_intervals = np.diff(annotation.sample) / record.fs  # sample rate from the record
# Calculate heart rate (in bpm)
heart_rate = 60 / rr_intervals
plt.figure(figsize=(10, 4))
plt.plot(heart_rate)
plt.title("Heart Rate over Time")
plt.xlabel("Time (samples)")
plt.ylabel("Heart Rate (bpm)")
plt.show()


# Calculate HRV (standard deviation of RR intervals)
hrv = np.std(rr_intervals)
print(f"Heart Rate Variability (HRV): {hrv} seconds")


#  Feature Engineering for Machine Learning
from scipy.stats import skew, kurtosis

features = {
    'mean': np.mean(rr_intervals),
    'std': np.std(rr_intervals),
    'skewness': skew(rr_intervals),
    'kurtosis': kurtosis(rr_intervals),
}
print(features)

# Example: Machine Learning Classification (Biometric Authentication)
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis

# Example: Assuming we have more than one record
# Load multiple records from we dataset (we may need to adjust paths based on our dataset)
records = [
    'ecg-id-database-1.0.0/Person_08/rec_2',
    'ecg-id-database-1.0.0/Person_01/rec_3',  # Add more records if available
]

# Store features and labels
features = []
labels = []

# Loop through multiple records
for record_path in records:
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # Extract ECG signal (assuming single lead ECG)
    ecg_signal = record.p_signal[:, 0]  # First channel (lead)
    
    # Calculate statistical features (e.g., mean, skewness, kurtosis)
    mean_val = np.mean(ecg_signal)
    skewness_val = skew(ecg_signal)
    kurtosis_val = kurtosis(ecg_signal)
    
    # Add features to the list
    features.append([mean_val, skewness_val, kurtosis_val])
    
    # For simplicity, assuming '0' for normal and '1' for abnormal (we can define this based on our annotations)
    labels.append(0)  # Just as an example; we might need to extract real labels from annotations

# Convert features and labels into numpy arrays
features = np.array(features)
labels = np.array(labels)

# Check the number of samples
print(f"Number of samples: {features.shape[0]}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# Loop through records and plot ECG signals
for record_path in records:
    record = wfdb.rdrecord(record_path)
    
    # Extract ECG signal (assuming single lead ECG)
    ecg_signal = record.p_signal[:, 0]  # First channel (lead)

    # Create a time array
    sampling_rate = record.fs
    time = np.arange(0, len(ecg_signal)) / sampling_rate

    # Plot the ECG signal
    plt.figure(figsize=(10, 6))
    plt.plot(time, ecg_signal)
    plt.title(f'ECG Signal - {record_path}')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
# Scatter plot to visualize the features
plt.figure(figsize=(10, 6))

# Scatter plot: Mean vs Skewness
plt.subplot(1, 2, 1)
plt.scatter(features[:, 0], features[:, 1], color=['red' if label == 0 else 'blue' for label in labels], label='Data points')

plt.title('Feature Visualization: Mean vs Skewness')
plt.xlabel('Mean')
plt.ylabel('Skewness')
plt.colorbar(label='Labels')  # Color indicates the class (normal/abnormal)
plt.grid(True)

# Scatter plot: Skewness vs Kurtosis
plt.subplot(1, 2, 2)
plt.scatter(features[:, 1], features[:, 2], color=['red' if label == 0 else 'blue' for label in labels], label='Data points')
plt.title('Feature Visualization: Skewness vs Kurtosis')
plt.xlabel('Skewness')
plt.ylabel('Kurtosis')
plt.colorbar(label='Labels')  # Color indicates the class (normal/abnormal)
plt.grid(True)

plt.tight_layout()
plt.show()


# Anamoly Detection using Isolation Forest
# Load the ECG signal and annotations
record_path = 'ecg-id-database-1.0.0/Person_90/rec_2'  # Adjust based on your record
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

# Extract ECG signal (assuming single lead ECG)
ecg_signal = record.p_signal[:, 0]  # First lead (if it's a multi-lead signal)

# Find R-peaks from annotations
r_peak_indices = annotation.sample  # Indices of R-peaks in the ECG signal

# Compute RR intervals (differences between consecutive R-peaks)
rr_intervals = np.diff(r_peak_indices) / record.fs  # In seconds (divide by sampling frequency)

# Apply Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.1)  # Adjust contamination parameter based on your data
anomalies = model.fit_predict(rr_intervals.reshape(-1, 1))  # Reshape to 2D for the model

# Plot RR intervals and anomalies
plt.figure(figsize=(10, 4))
plt.plot(rr_intervals, label="RR Intervals")
plt.scatter(np.where(anomalies == -1), rr_intervals[anomalies == -1], color='red', label='Anomalies')
plt.title("Anomaly Detection on RR Intervals")
plt.xlabel("Time (samples)")
plt.ylabel("RR Interval (seconds)")
plt.legend()
plt.show()

