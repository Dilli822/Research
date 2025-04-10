import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet  # Use MobileNetV1 instead of V2
from tensorflow.keras.optimizers import Adam
import numpy as np
import wfdb
from scipy.signal import spectrogram
import os

# Function to generate records for Person_01 to Person_90 with rec_1
def generate_records_for_persons():
    records = []
    base_path = 'ecg-id-database-1.0.0/'

    # Loop over persons 1 to 70 (training set)
    for i in range(1, 91):  # From Person_01 to Person_70
        person_id = f"Person_{i:02d}"
        rec_path = os.path.join(base_path, person_id, "rec_1")
        records.append(rec_path)
    
    return records

# Generate the records dynamically
records = generate_records_for_persons()

# Load ECG signal
def load_ecg_signal(file_path):
    try:
        record = wfdb.rdrecord(file_path)
        signal = record.p_signal[:, 0]  # Assuming single-channel ECG
        return signal
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")

# Convert ECG to spectrogram
def ecg_to_spectrogram(signal, fs=256, nperseg=64):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
    return np.log(Sxx + 1e-6)  # Log scale

# Preprocess data
def preprocess_data(records):
    spectrograms = []
    labels = []
    for record in records:
        signal = load_ecg_signal(record)
        spectrogram_data = ecg_to_spectrogram(signal)
      
        # Resize spectrogram to (224, 224) for MobileNetV1
        spectrogram_resized = tf.image.resize(
            np.expand_dims(spectrogram_data, axis=-1),  # Add channel dim
            (224, 224)
        ).numpy()
      
        # Repeat grayscale to 3 channels (RGB)
        spectrogram_rgb = np.repeat(spectrogram_resized, 3, axis=-1)
      
        spectrograms.append(spectrogram_rgb)
        labels.append(1 if "Person_08" in record else 0)  # Example label (Authenticated for Person_08)
  
    return np.array(spectrograms), np.array(labels)

# Create MobileNetV1 + GRU model
def create_mobilenet_gru_model(input_shape=(224, 224, 3)):
    # Load MobileNetV1 (without top layers)
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze weights

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Reshape((-1, 1024)),  # MobileNetV1 output is (7,7,1024) after pooling
        layers.GRU(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
  
    return model

# Preprocess training data
X_train, y_train = preprocess_data(records)
X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train)) * 2 - 1
print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")

# Create and compile model
model = create_mobilenet_gru_model()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.summary() # add this line.

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Function to generate test records for Person_71 to Person_90, excluding Person_08
def generate_test_records_for_persons():
    records = []
    base_path = 'ecg-id-database-1.0.0/'

    # Loop over persons 71 to 90 (test set)
    for i in range(5, 11):  # From Person_71 to Person_90
        person_id = f"Person_{i:02d}"
        rec_path = os.path.join(base_path, person_id, "rec_1")
        
        # Skip 'Person_08/rec_1'
        # if person_id == "Person_08" and "rec_1" in rec_path:
        #     continue
        
        records.append(rec_path)
    
    return records

# Generate the test records dynamically
test_records = generate_test_records_for_persons()

# Preprocess the test data
X_test, y_test = preprocess_data(test_records)
X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test)) * 2 - 1
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Train the model and capture the history
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Print history keys to see available metrics
print(history.history.keys())

import matplotlib.pyplot as plt

# Plot Training & Validation Loss
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Predict for each test record
for i, test_record in enumerate(test_records):
    # Preprocess the test record
    signal = load_ecg_signal(test_record)
    spectrogram_data = ecg_to_spectrogram(signal)
    
    # Resize and repeat for RGB channels
    spectrogram_resized = tf.image.resize(np.expand_dims(spectrogram_data, axis=-1), (224, 224)).numpy()
    spectrogram_rgb = np.repeat(spectrogram_resized, 3, axis=-1)
    
    # Normalize the input
    spectrogram_rgb = (spectrogram_rgb - np.min(spectrogram_rgb)) / (np.max(spectrogram_rgb) - np.min(spectrogram_rgb)) * 2 - 1
    
    # Predict using the model
    prediction = model.predict(np.expand_dims(spectrogram_rgb, axis=0))  # Add batch dimension
    predicted_class = "Accepted" if prediction >= 0.5 else "Not Accepted"
    
    # Print prediction and accuracy
    true_label = 1 if "Person_08" in test_record else 0  # Assuming label 1 for Person_08
    print(f"Test Record {test_record}: Prediction = {predicted_class}, Actual = {'Accepted' if true_label == 1 else 'Not Accepted'}, Accuracy = {prediction[0][0]:.4f}")

# Test genuine user (Person_08/rec_1 or any other file from Person_08)
# Recompute min and max from the training data
train_min, train_max = np.min(X_train), np.max(X_train)

# Test genuine user (Person_08/rec_1 or any other file from Person_08)
genuine_user_path = 'ecg-id-database-1.0.0/Person_08/rec_1'
signal = load_ecg_signal(genuine_user_path)
spectrogram_data = ecg_to_spectrogram(signal)
spectrogram_resized = tf.image.resize(np.expand_dims(spectrogram_data, -1), (224, 224)).numpy()
spectrogram_rgb = np.repeat(spectrogram_resized, 3, -1)
X_input = np.expand_dims(spectrogram_rgb, 0)

# Normalize with the recomputed min and max values
X_input = (X_input - train_min) / (train_max - train_min) * 2 - 1

prediction = model.predict(X_input, verbose=0)[0][0]
predicted_class = "Accepted" if prediction >= 0.5 else "Not Accepted"
print(f"Genuine User Test: Prediction = {predicted_class}, Probability = {prediction:.4f}")
