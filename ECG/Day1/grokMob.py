# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import numpy as np
# import wfdb
# from scipy.signal import spectrogram
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import os


# # -----------------------
# # Load ECG Signal Function
# # -----------------------
# def load_ecg_signal(record_path):
#    record = wfdb.rdrecord(record_path)
#    signal = record.p_signal[:, 0]  # Use first channel
#    return signal


# # -----------------------
# # Convert ECG to Spectrogram
# # -----------------------
# def ecg_to_spectrogram(signal, fs=256, nperseg=64):
#    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
#    log_spec = np.log(Sxx + 1e-6)  # Avoid log(0)
#    return log_spec


# # -----------------------
# # Preprocess ECG Data
# # -----------------------
# def preprocess_data(records, positive_label="Person_01"):
#    spectrograms = []
#    labels = []


#    for record in records:
#        signal = load_ecg_signal(record)
#        spec = ecg_to_spectrogram(signal)
#        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-6)
#        spec_resized = tf.image.resize(np.expand_dims(spec, axis=-1), (224, 224)).numpy()
#        spec_rgb = np.repeat(spec_resized, 3, axis=-1)


#        spectrograms.append(spec_rgb)
#        label = 1 if positive_label in record else 0
#        labels.append(label)


#    return np.array(spectrograms), np.array(labels)


# # -----------------------
# # Create MobileNet + GRU Model
# # -----------------------
# def create_mobilenet_gru_model(input_shape=(224, 224, 3)):
#    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
  
#    # Fine-tune last 20 layers
#    base_model.trainable = True
#    for layer in base_model.layers[:-20]:
#        layer.trainable = False


#    model = models.Sequential([
#        base_model,
#        layers.GlobalAveragePooling2D(),
#        layers.Reshape((1, 1024)),
#        layers.GRU(64, return_sequences=False, dropout=0.3),  # Added dropout
#        layers.Dense(32, activation='relu'),
#        layers.Dropout(0.3),
#        layers.Dense(1, activation='sigmoid')
#    ])
#    return model


# # -----------------------
# # Load and Preprocess Data
# # -----------------------
# train_records = [
# 'ecg-id-database-1.0.0/Person_01/rec_1',
# 'ecg-id-database-1.0.0/Person_02/rec_1',
# 'ecg-id-database-1.0.0/Person_03/rec_1',
# 'ecg-id-database-1.0.0/Person_04/rec_1',
# 'ecg-id-database-1.0.0/Person_05/rec_1',
# 'ecg-id-database-1.0.0/Person_06/rec_1',
# 'ecg-id-database-1.0.0/Person_07/rec_1',
# 'ecg-id-database-1.0.0/Person_08/rec_1',
# 'ecg-id-database-1.0.0/Person_09/rec_1',
# 'ecg-id-database-1.0.0/Person_10/rec_1',
# 'ecg-id-database-1.0.0/Person_11/rec_1',
# 'ecg-id-database-1.0.0/Person_12/rec_1',
# 'ecg-id-database-1.0.0/Person_13/rec_1',
# 'ecg-id-database-1.0.0/Person_14/rec_1',
# 'ecg-id-database-1.0.0/Person_15/rec_1',
# 'ecg-id-database-1.0.0/Person_16/rec_1',
# 'ecg-id-database-1.0.0/Person_17/rec_1',
# 'ecg-id-database-1.0.0/Person_18/rec_1',
# 'ecg-id-database-1.0.0/Person_19/rec_1',
# 'ecg-id-database-1.0.0/Person_20/rec_1',
# 'ecg-id-database-1.0.0/Person_21/rec_1',
# 'ecg-id-database-1.0.0/Person_22/rec_1',
# 'ecg-id-database-1.0.0/Person_23/rec_1',
# 'ecg-id-database-1.0.0/Person_24/rec_1',
# 'ecg-id-database-1.0.0/Person_25/rec_1',
# 'ecg-id-database-1.0.0/Person_26/rec_1',
# 'ecg-id-database-1.0.0/Person_27/rec_1',
# 'ecg-id-database-1.0.0/Person_28/rec_1',
# 'ecg-id-database-1.0.0/Person_29/rec_1',
# 'ecg-id-database-1.0.0/Person_30/rec_1',
# 'ecg-id-database-1.0.0/Person_31/rec_1',
# 'ecg-id-database-1.0.0/Person_32/rec_1',
# 'ecg-id-database-1.0.0/Person_33/rec_1',
# 'ecg-id-database-1.0.0/Person_34/rec_1',
# 'ecg-id-database-1.0.0/Person_35/rec_1',
# 'ecg-id-database-1.0.0/Person_36/rec_1',
# 'ecg-id-database-1.0.0/Person_37/rec_1',
# 'ecg-id-database-1.0.0/Person_38/rec_1',
# 'ecg-id-database-1.0.0/Person_39/rec_1',
# 'ecg-id-database-1.0.0/Person_40/rec_1',
# 'ecg-id-database-1.0.0/Person_41/rec_1',
# 'ecg-id-database-1.0.0/Person_42/rec_1',
# 'ecg-id-database-1.0.0/Person_43/rec_1',
# 'ecg-id-database-1.0.0/Person_44/rec_1',
# 'ecg-id-database-1.0.0/Person_45/rec_1',
# 'ecg-id-database-1.0.0/Person_46/rec_1',
# 'ecg-id-database-1.0.0/Person_47/rec_1',
# 'ecg-id-database-1.0.0/Person_48/rec_1',
# 'ecg-id-database-1.0.0/Person_49/rec_1',
# 'ecg-id-database-1.0.0/Person_50/rec_1',
# 'ecg-id-database-1.0.0/Person_51/rec_1',
# 'ecg-id-database-1.0.0/Person_52/rec_1',
# 'ecg-id-database-1.0.0/Person_53/rec_1',
# 'ecg-id-database-1.0.0/Person_54/rec_1',
# 'ecg-id-database-1.0.0/Person_55/rec_1',
# 'ecg-id-database-1.0.0/Person_56/rec_1',
# 'ecg-id-database-1.0.0/Person_57/rec_1',
# 'ecg-id-database-1.0.0/Person_58/rec_1',
# 'ecg-id-database-1.0.0/Person_59/rec_1',
# 'ecg-id-database-1.0.0/Person_60/rec_1',
# 'ecg-id-database-1.0.0/Person_61/rec_1',
# 'ecg-id-database-1.0.0/Person_62/rec_1',
# 'ecg-id-database-1.0.0/Person_63/rec_1',
# 'ecg-id-database-1.0.0/Person_64/rec_1',
# 'ecg-id-database-1.0.0/Person_65/rec_1',
# 'ecg-id-database-1.0.0/Person_66/rec_1',
# 'ecg-id-database-1.0.0/Person_67/rec_1',
# 'ecg-id-database-1.0.0/Person_68/rec_1',
# 'ecg-id-database-1.0.0/Person_69/rec_1',
# 'ecg-id-database-1.0.0/Person_70/rec_1',
# 'ecg-id-database-1.0.0/Person_71/rec_1',
# 'ecg-id-database-1.0.0/Person_72/rec_1',
# 'ecg-id-database-1.0.0/Person_73/rec_1',
# 'ecg-id-database-1.0.0/Person_74/rec_1',
# 'ecg-id-database-1.0.0/Person_75/rec_1',
# 'ecg-id-database-1.0.0/Person_76/rec_1',
# 'ecg-id-database-1.0.0/Person_77/rec_1',
# 'ecg-id-database-1.0.0/Person_78/rec_1',
# 'ecg-id-database-1.0.0/Person_79/rec_1',
# 'ecg-id-database-1.0.0/Person_80/rec_1',
# 'ecg-id-database-1.0.0/Person_81/rec_1',
# 'ecg-id-database-1.0.0/Person_82/rec_1',
# 'ecg-id-database-1.0.0/Person_83/rec_1',
# 'ecg-id-database-1.0.0/Person_84/rec_1',
# 'ecg-id-database-1.0.0/Person_85/rec_1',
# 'ecg-id-database-1.0.0/Person_86/rec_1',
# 'ecg-id-database-1.0.0/Person_87/rec_1',
# 'ecg-id-database-1.0.0/Person_88/rec_1',
# 'ecg-id-database-1.0.0/Person_89/rec_1',
# 'ecg-id-database-1.0.0/Person_90/rec_1',

# ]


# X_train, y_train = preprocess_data(train_records)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


# print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")


# # -----------------------
# # Compile and Train Model
# # -----------------------
# model = create_mobilenet_gru_model()
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()


# # -----------------------
# # Evaluate on Test Records
# # -----------------------
# test_records = [
# 'ecg-id-database-1.0.0/Person_01/rec_1',
# 'ecg-id-database-1.0.0/Person_72/rec_8',
# 'ecg-id-database-1.0.0/Person_73/rec_1',
# 'ecg-id-database-1.0.0/Person_13/rec_1',
# ]

# print(f"Total test records: {len(test_records)}")
# total_loss = 0
# total_accuracy = 0

# for i, record in enumerate(test_records):
#    X_test, y_test = preprocess_data([record])
#    X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test)) * 2 - 1
#    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#    print(f"Record {i+1}/{len(test_records)} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#    print("Status: Authenticated\n" if accuracy >= 0.9 else "Status: Not Authenticated\n")
#    total_loss += loss
#    total_accuracy += accuracy


# avg_loss = total_loss / len(test_records)
# avg_accuracy = total_accuracy / len(test_records)
# print(f"Average Test Loss: {avg_loss:.4f}, Average Test Accuracy: {avg_accuracy:.4f}")


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import wfdb
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


# -----------------------
# Load ECG Signal Function
# -----------------------
def load_ecg_signal(record_path):
   record = wfdb.rdrecord(record_path)
   signal = record.p_signal[:, 0]  # Use first channel
   return signal


# -----------------------
# Convert ECG to Spectrogram
# -----------------------
def ecg_to_spectrogram(signal, fs=256, nperseg=64):
   f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
   log_spec = np.log(Sxx + 1e-6)  # Avoid log(0)
   return log_spec


# -----------------------
# Preprocess ECG Data
# -----------------------
def preprocess_data(records, positive_label="Person_01"):
   spectrograms = []
   labels = []

   for record in records:
       signal = load_ecg_signal(record)
       spec = ecg_to_spectrogram(signal)
       spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-6)
       spec_resized = tf.image.resize(np.expand_dims(spec, axis=-1), (224, 224)).numpy()
       spec_rgb = np.repeat(spec_resized, 3, axis=-1)

       spectrograms.append(spec_rgb)
       label = 1 if positive_label in record else 0
       labels.append(label)

   return np.array(spectrograms), np.array(labels)


# -----------------------
# Create MobileNet + GRU Model
# -----------------------
def create_mobilenet_gru_model(input_shape=(224, 224, 3)):
   base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
  
   # Fine-tune last 50 layers (instead of 20)
   base_model.trainable = True
   for layer in base_model.layers[:-50]:
       layer.trainable = False

   model = models.Sequential([
       base_model,
       layers.GlobalAveragePooling2D(),
       layers.Reshape((1, 1024)),
       layers.GRU(64, return_sequences=False, dropout=0.3),  # Added dropout
       layers.Dense(32, activation='relu'),
       layers.Dropout(0.3),
       layers.Dense(1, activation='sigmoid')
   ])
   return model

# -----------------------
# Load and Preprocess Data
# -----------------------
train_records = [
'ecg-id-database-1.0.0/Person_01/rec_1',
'ecg-id-database-1.0.0/Person_02/rec_1',
'ecg-id-database-1.0.0/Person_03/rec_1',
'ecg-id-database-1.0.0/Person_04/rec_1',
'ecg-id-database-1.0.0/Person_05/rec_1',
'ecg-id-database-1.0.0/Person_06/rec_1',
'ecg-id-database-1.0.0/Person_07/rec_1',
'ecg-id-database-1.0.0/Person_08/rec_1',
'ecg-id-database-1.0.0/Person_09/rec_1',
'ecg-id-database-1.0.0/Person_10/rec_1',
'ecg-id-database-1.0.0/Person_11/rec_1',
'ecg-id-database-1.0.0/Person_12/rec_1',
'ecg-id-database-1.0.0/Person_13/rec_1',
'ecg-id-database-1.0.0/Person_14/rec_1',
'ecg-id-database-1.0.0/Person_15/rec_1',
'ecg-id-database-1.0.0/Person_16/rec_1',
'ecg-id-database-1.0.0/Person_17/rec_1',
'ecg-id-database-1.0.0/Person_18/rec_1',
'ecg-id-database-1.0.0/Person_19/rec_1',
'ecg-id-database-1.0.0/Person_20/rec_1',
'ecg-id-database-1.0.0/Person_21/rec_1',
'ecg-id-database-1.0.0/Person_22/rec_1',
'ecg-id-database-1.0.0/Person_23/rec_1',
'ecg-id-database-1.0.0/Person_24/rec_1',
'ecg-id-database-1.0.0/Person_25/rec_1',
'ecg-id-database-1.0.0/Person_26/rec_1',
'ecg-id-database-1.0.0/Person_27/rec_1',
'ecg-id-database-1.0.0/Person_28/rec_1',
'ecg-id-database-1.0.0/Person_29/rec_1',
'ecg-id-database-1.0.0/Person_30/rec_1',
'ecg-id-database-1.0.0/Person_31/rec_1',
'ecg-id-database-1.0.0/Person_32/rec_1',
'ecg-id-database-1.0.0/Person_33/rec_1',
'ecg-id-database-1.0.0/Person_34/rec_1',
'ecg-id-database-1.0.0/Person_35/rec_1',
'ecg-id-database-1.0.0/Person_36/rec_1',
'ecg-id-database-1.0.0/Person_37/rec_1',
'ecg-id-database-1.0.0/Person_38/rec_1',
'ecg-id-database-1.0.0/Person_39/rec_1',
'ecg-id-database-1.0.0/Person_40/rec_1',
'ecg-id-database-1.0.0/Person_41/rec_1',
'ecg-id-database-1.0.0/Person_42/rec_1',
'ecg-id-database-1.0.0/Person_43/rec_1',
'ecg-id-database-1.0.0/Person_44/rec_1',
'ecg-id-database-1.0.0/Person_45/rec_1',
'ecg-id-database-1.0.0/Person_46/rec_1',
'ecg-id-database-1.0.0/Person_47/rec_1',
'ecg-id-database-1.0.0/Person_48/rec_1',
'ecg-id-database-1.0.0/Person_49/rec_1',
'ecg-id-database-1.0.0/Person_50/rec_1',
'ecg-id-database-1.0.0/Person_51/rec_1',
'ecg-id-database-1.0.0/Person_52/rec_1',
'ecg-id-database-1.0.0/Person_53/rec_1',
'ecg-id-database-1.0.0/Person_54/rec_1',
'ecg-id-database-1.0.0/Person_55/rec_1',
'ecg-id-database-1.0.0/Person_56/rec_1',
'ecg-id-database-1.0.0/Person_57/rec_1',
'ecg-id-database-1.0.0/Person_58/rec_1',
'ecg-id-database-1.0.0/Person_59/rec_1',
'ecg-id-database-1.0.0/Person_60/rec_1',
'ecg-id-database-1.0.0/Person_61/rec_1',
'ecg-id-database-1.0.0/Person_62/rec_1',
'ecg-id-database-1.0.0/Person_63/rec_1',
'ecg-id-database-1.0.0/Person_64/rec_1',
'ecg-id-database-1.0.0/Person_65/rec_1',
'ecg-id-database-1.0.0/Person_66/rec_1',
'ecg-id-database-1.0.0/Person_67/rec_1',
'ecg-id-database-1.0.0/Person_68/rec_1',
'ecg-id-database-1.0.0/Person_69/rec_1',
'ecg-id-database-1.0.0/Person_70/rec_1',
'ecg-id-database-1.0.0/Person_71/rec_1',
'ecg-id-database-1.0.0/Person_72/rec_1',
'ecg-id-database-1.0.0/Person_73/rec_1',
'ecg-id-database-1.0.0/Person_74/rec_1',
'ecg-id-database-1.0.0/Person_75/rec_1',
'ecg-id-database-1.0.0/Person_76/rec_1',
'ecg-id-database-1.0.0/Person_77/rec_1',
'ecg-id-database-1.0.0/Person_78/rec_1',
'ecg-id-database-1.0.0/Person_79/rec_1',
'ecg-id-database-1.0.0/Person_80/rec_1',
'ecg-id-database-1.0.0/Person_81/rec_1',
'ecg-id-database-1.0.0/Person_82/rec_1',
'ecg-id-database-1.0.0/Person_83/rec_1',
'ecg-id-database-1.0.0/Person_84/rec_1',
'ecg-id-database-1.0.0/Person_85/rec_1',
'ecg-id-database-1.0.0/Person_86/rec_1',
'ecg-id-database-1.0.0/Person_87/rec_1',
'ecg-id-database-1.0.0/Person_88/rec_1',
'ecg-id-database-1.0.0/Person_89/rec_1',
'ecg-id-database-1.0.0/Person_90/rec_1',
]


X_train, y_train = preprocess_data(train_records)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Normalize data
X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train) + 1e-6)
X_val = (X_val - np.min(X_val)) / (np.max(X_val) - np.min(X_val) + 1e-6)

print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")


# -----------------------
# Compile and Train Model
# -----------------------
model = create_mobilenet_gru_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# -----------------------
# Evaluate on Test Records
# -----------------------
test_records = [
    'ecg-id-database-1.0.0/Person_01/rec_1',
    'ecg-id-database-1.0.0/Person_72/rec_8',
    # Add other test records
]

print(f"Total test records: {len(test_records)}")
total_loss = 0
total_accuracy = 0
correct_predictions = 0
incorrect_predictions = 0

for i, record in enumerate(test_records):
    X_test, y_test = preprocess_data([record])
    
    # Normalize the test data the same way as training data
    # X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test) + 1e-6)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Apply threshold for classification (0.5)
    predicted_class = 1 if accuracy >= 0.5 else 0
    
    print(f"Record {i+1}/{len(test_records)} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    print("Status: Authenticated\n" if predicted_class == y_test else "Status: Not Authenticated\n")
    
    total_loss += loss
    total_accuracy += accuracy
    
    if predicted_class == y_test:
        correct_predictions += 1
    else:
        incorrect_predictions += 1

# Calculate average metrics
avg_loss = total_loss / len(test_records)
avg_accuracy = total_accuracy / len(test_records)

print(f"Average Test Loss: {avg_loss:.4f}, Average Test Accuracy: {avg_accuracy:.4f}")
print(f"Correct Predictions: {correct_predictions}/{len(test_records)}")
print(f"Incorrect Predictions: {incorrect_predictions}/{len(test_records)}")