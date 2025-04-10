import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
import numpy as np
import wfdb
from scipy.signal import spectrogram

# Define train_records (same as yours, omitted for brevity)
train_records = [
'ecg-id-database-1.0.0/Person_01/rec_1',
'ecg-id-database-1.0.0/Person_01/rec_10',
'ecg-id-database-1.0.0/Person_01/rec_11',
'ecg-id-database-1.0.0/Person_01/rec_12',
'ecg-id-database-1.0.0/Person_01/rec_13',
'ecg-id-database-1.0.0/Person_01/rec_14',
'ecg-id-database-1.0.0/Person_01/rec_15',
'ecg-id-database-1.0.0/Person_01/rec_16',
'ecg-id-database-1.0.0/Person_01/rec_17',
'ecg-id-database-1.0.0/Person_01/rec_18',
'ecg-id-database-1.0.0/Person_01/rec_19',
'ecg-id-database-1.0.0/Person_01/rec_2',
'ecg-id-database-1.0.0/Person_01/rec_20',
'ecg-id-database-1.0.0/Person_01/rec_3',
'ecg-id-database-1.0.0/Person_01/rec_4',
'ecg-id-database-1.0.0/Person_01/rec_5',
'ecg-id-database-1.0.0/Person_01/rec_6',
'ecg-id-database-1.0.0/Person_01/rec_7',
'ecg-id-database-1.0.0/Person_01/rec_8',
'ecg-id-database-1.0.0/Person_01/rec_9',
'ecg-id-database-1.0.0/Person_02/rec_1',
'ecg-id-database-1.0.0/Person_02/rec_10',
'ecg-id-database-1.0.0/Person_02/rec_11',
'ecg-id-database-1.0.0/Person_02/rec_12',
'ecg-id-database-1.0.0/Person_02/rec_13',
'ecg-id-database-1.0.0/Person_02/rec_14',
'ecg-id-database-1.0.0/Person_02/rec_15',
'ecg-id-database-1.0.0/Person_02/rec_16',
'ecg-id-database-1.0.0/Person_02/rec_17',
'ecg-id-database-1.0.0/Person_02/rec_18',
'ecg-id-database-1.0.0/Person_02/rec_19',
'ecg-id-database-1.0.0/Person_02/rec_2',
'ecg-id-database-1.0.0/Person_02/rec_20',
'ecg-id-database-1.0.0/Person_02/rec_21',
'ecg-id-database-1.0.0/Person_02/rec_22',
'ecg-id-database-1.0.0/Person_02/rec_3',
'ecg-id-database-1.0.0/Person_02/rec_4',
'ecg-id-database-1.0.0/Person_02/rec_5',
'ecg-id-database-1.0.0/Person_02/rec_6',
'ecg-id-database-1.0.0/Person_02/rec_7',
'ecg-id-database-1.0.0/Person_02/rec_8',
'ecg-id-database-1.0.0/Person_02/rec_9',
'ecg-id-database-1.0.0/Person_03/rec_1',
'ecg-id-database-1.0.0/Person_03/rec_2',
'ecg-id-database-1.0.0/Person_03/rec_3',
'ecg-id-database-1.0.0/Person_03/rec_4',
'ecg-id-database-1.0.0/Person_03/rec_5',
'ecg-id-database-1.0.0/Person_04/rec_1',
'ecg-id-database-1.0.0/Person_04/rec_2',
'ecg-id-database-1.0.0/Person_05/rec_1',
'ecg-id-database-1.0.0/Person_05/rec_2',
'ecg-id-database-1.0.0/Person_06/rec_1',
'ecg-id-database-1.0.0/Person_06/rec_2',
'ecg-id-database-1.0.0/Person_07/rec_1',
'ecg-id-database-1.0.0/Person_07/rec_2',
'ecg-id-database-1.0.0/Person_08/rec_1',
'ecg-id-database-1.0.0/Person_08/rec_2',
'ecg-id-database-1.0.0/Person_09/rec_1',
'ecg-id-database-1.0.0/Person_09/rec_2',
'ecg-id-database-1.0.0/Person_09/rec_3',
'ecg-id-database-1.0.0/Person_09/rec_4',
'ecg-id-database-1.0.0/Person_09/rec_5',
'ecg-id-database-1.0.0/Person_09/rec_6',
'ecg-id-database-1.0.0/Person_09/rec_7',
'ecg-id-database-1.0.0/Person_10/rec_1',
'ecg-id-database-1.0.0/Person_10/rec_2',
'ecg-id-database-1.0.0/Person_10/rec_3',
'ecg-id-database-1.0.0/Person_10/rec_4',
'ecg-id-database-1.0.0/Person_10/rec_5',
'ecg-id-database-1.0.0/Person_11/rec_1',
'ecg-id-database-1.0.0/Person_11/rec_2',
'ecg-id-database-1.0.0/Person_11/rec_3',
'ecg-id-database-1.0.0/Person_12/rec_1',
'ecg-id-database-1.0.0/Person_12/rec_2',
'ecg-id-database-1.0.0/Person_13/rec_1',
'ecg-id-database-1.0.0/Person_13/rec_2',
'ecg-id-database-1.0.0/Person_14/rec_1',
'ecg-id-database-1.0.0/Person_14/rec_2',
'ecg-id-database-1.0.0/Person_14/rec_3',
'ecg-id-database-1.0.0/Person_15/rec_1',
'ecg-id-database-1.0.0/Person_15/rec_2',
'ecg-id-database-1.0.0/Person_16/rec_1',
'ecg-id-database-1.0.0/Person_16/rec_2',
'ecg-id-database-1.0.0/Person_16/rec_3',
'ecg-id-database-1.0.0/Person_17/rec_1',
'ecg-id-database-1.0.0/Person_17/rec_2',
'ecg-id-database-1.0.0/Person_18/rec_1',
'ecg-id-database-1.0.0/Person_18/rec_2',
'ecg-id-database-1.0.0/Person_19/rec_1',
'ecg-id-database-1.0.0/Person_19/rec_2',
'ecg-id-database-1.0.0/Person_20/rec_1',
'ecg-id-database-1.0.0/Person_20/rec_2',
'ecg-id-database-1.0.0/Person_21/rec_1',
'ecg-id-database-1.0.0/Person_21/rec_2',
'ecg-id-database-1.0.0/Person_21/rec_3',
'ecg-id-database-1.0.0/Person_22/rec_1',
'ecg-id-database-1.0.0/Person_22/rec_2',
'ecg-id-database-1.0.0/Person_23/rec_1',
'ecg-id-database-1.0.0/Person_23/rec_2',
'ecg-id-database-1.0.0/Person_24/rec_1',
'ecg-id-database-1.0.0/Person_24/rec_2',
'ecg-id-database-1.0.0/Person_24/rec_3',
'ecg-id-database-1.0.0/Person_24/rec_4',
'ecg-id-database-1.0.0/Person_24/rec_5',
'ecg-id-database-1.0.0/Person_25/rec_1',
'ecg-id-database-1.0.0/Person_25/rec_2',
'ecg-id-database-1.0.0/Person_25/rec_3',
'ecg-id-database-1.0.0/Person_25/rec_4',
'ecg-id-database-1.0.0/Person_25/rec_5',
'ecg-id-database-1.0.0/Person_26/rec_1',
'ecg-id-database-1.0.0/Person_26/rec_2',
'ecg-id-database-1.0.0/Person_26/rec_3',
'ecg-id-database-1.0.0/Person_26/rec_4',
'ecg-id-database-1.0.0/Person_27/rec_1',
'ecg-id-database-1.0.0/Person_27/rec_2',
'ecg-id-database-1.0.0/Person_27/rec_3',
'ecg-id-database-1.0.0/Person_28/rec_1',
'ecg-id-database-1.0.0/Person_28/rec_2',
'ecg-id-database-1.0.0/Person_28/rec_3',
'ecg-id-database-1.0.0/Person_28/rec_4',
'ecg-id-database-1.0.0/Person_28/rec_5',
'ecg-id-database-1.0.0/Person_29/rec_1',
'ecg-id-database-1.0.0/Person_29/rec_2',
'ecg-id-database-1.0.0/Person_30/rec_1',
'ecg-id-database-1.0.0/Person_30/rec_2',
'ecg-id-database-1.0.0/Person_30/rec_3',
'ecg-id-database-1.0.0/Person_30/rec_4',
'ecg-id-database-1.0.0/Person_30/rec_5',
'ecg-id-database-1.0.0/Person_31/rec_1',
'ecg-id-database-1.0.0/Person_31/rec_2',
'ecg-id-database-1.0.0/Person_32/rec_1',
'ecg-id-database-1.0.0/Person_32/rec_2',
'ecg-id-database-1.0.0/Person_32/rec_3',
'ecg-id-database-1.0.0/Person_32/rec_4',
'ecg-id-database-1.0.0/Person_32/rec_5',
'ecg-id-database-1.0.0/Person_32/rec_6',
'ecg-id-database-1.0.0/Person_33/rec_1',
'ecg-id-database-1.0.0/Person_33/rec_2',
'ecg-id-database-1.0.0/Person_34/rec_1',
'ecg-id-database-1.0.0/Person_34/rec_2',
'ecg-id-database-1.0.0/Person_34/rec_3',
'ecg-id-database-1.0.0/Person_34/rec_4',
'ecg-id-database-1.0.0/Person_34/rec_5',
'ecg-id-database-1.0.0/Person_35/rec_1',
'ecg-id-database-1.0.0/Person_35/rec_2',
'ecg-id-database-1.0.0/Person_35/rec_3',
'ecg-id-database-1.0.0/Person_35/rec_4',
'ecg-id-database-1.0.0/Person_35/rec_5',
'ecg-id-database-1.0.0/Person_36/rec_1',
'ecg-id-database-1.0.0/Person_36/rec_2',
'ecg-id-database-1.0.0/Person_36/rec_3',
'ecg-id-database-1.0.0/Person_36/rec_4',
'ecg-id-database-1.0.0/Person_36/rec_5',
'ecg-id-database-1.0.0/Person_37/rec_1',
'ecg-id-database-1.0.0/Person_37/rec_2',
'ecg-id-database-1.0.0/Person_38/rec_1',
'ecg-id-database-1.0.0/Person_38/rec_2',
'ecg-id-database-1.0.0/Person_39/rec_1',
'ecg-id-database-1.0.0/Person_39/rec_2',
'ecg-id-database-1.0.0/Person_40/rec_1',
'ecg-id-database-1.0.0/Person_40/rec_2',
'ecg-id-database-1.0.0/Person_40/rec_3',
'ecg-id-database-1.0.0/Person_40/rec_4',
'ecg-id-database-1.0.0/Person_41/rec_1',
'ecg-id-database-1.0.0/Person_41/rec_2',
'ecg-id-database-1.0.0/Person_42/rec_1',
'ecg-id-database-1.0.0/Person_42/rec_2',
'ecg-id-database-1.0.0/Person_42/rec_3',
'ecg-id-database-1.0.0/Person_42/rec_4',
'ecg-id-database-1.0.0/Person_43/rec_1',
'ecg-id-database-1.0.0/Person_43/rec_2',
'ecg-id-database-1.0.0/Person_44/rec_1',
'ecg-id-database-1.0.0/Person_44/rec_2',
'ecg-id-database-1.0.0/Person_45/rec_1',
'ecg-id-database-1.0.0/Person_45/rec_2',
'ecg-id-database-1.0.0/Person_46/rec_1',
'ecg-id-database-1.0.0/Person_46/rec_2',
'ecg-id-database-1.0.0/Person_46/rec_3',
'ecg-id-database-1.0.0/Person_46/rec_4',
'ecg-id-database-1.0.0/Person_46/rec_5',
'ecg-id-database-1.0.0/Person_47/rec_1',
'ecg-id-database-1.0.0/Person_47/rec_2',
'ecg-id-database-1.0.0/Person_48/rec_1',
'ecg-id-database-1.0.0/Person_48/rec_2',
'ecg-id-database-1.0.0/Person_49/rec_1',
'ecg-id-database-1.0.0/Person_49/rec_2',
'ecg-id-database-1.0.0/Person_50/rec_1',
'ecg-id-database-1.0.0/Person_50/rec_2',
'ecg-id-database-1.0.0/Person_51/rec_1',
'ecg-id-database-1.0.0/Person_51/rec_2',
'ecg-id-database-1.0.0/Person_51/rec_3',
'ecg-id-database-1.0.0/Person_51/rec_4',
'ecg-id-database-1.0.0/Person_52/rec_1',
'ecg-id-database-1.0.0/Person_52/rec_10',
'ecg-id-database-1.0.0/Person_52/rec_11',
'ecg-id-database-1.0.0/Person_52/rec_2',
'ecg-id-database-1.0.0/Person_52/rec_3',
'ecg-id-database-1.0.0/Person_52/rec_4',
'ecg-id-database-1.0.0/Person_52/rec_5',
'ecg-id-database-1.0.0/Person_52/rec_6',
'ecg-id-database-1.0.0/Person_52/rec_7',
'ecg-id-database-1.0.0/Person_52/rec_8',
'ecg-id-database-1.0.0/Person_52/rec_9',
'ecg-id-database-1.0.0/Person_53/rec_1',
'ecg-id-database-1.0.0/Person_53/rec_2',
'ecg-id-database-1.0.0/Person_53/rec_3',
'ecg-id-database-1.0.0/Person_53/rec_4',
'ecg-id-database-1.0.0/Person_53/rec_5',
'ecg-id-database-1.0.0/Person_54/rec_1',
'ecg-id-database-1.0.0/Person_54/rec_2',
'ecg-id-database-1.0.0/Person_55/rec_1',
'ecg-id-database-1.0.0/Person_55/rec_2',
'ecg-id-database-1.0.0/Person_56/rec_1',
'ecg-id-database-1.0.0/Person_56/rec_2',
'ecg-id-database-1.0.0/Person_57/rec_1',
'ecg-id-database-1.0.0/Person_57/rec_2',
'ecg-id-database-1.0.0/Person_57/rec_3',
'ecg-id-database-1.0.0/Person_58/rec_1',
'ecg-id-database-1.0.0/Person_58/rec_2',
'ecg-id-database-1.0.0/Person_59/rec_1',
'ecg-id-database-1.0.0/Person_59/rec_2',
'ecg-id-database-1.0.0/Person_59/rec_3',
'ecg-id-database-1.0.0/Person_59/rec_4',
'ecg-id-database-1.0.0/Person_59/rec_5',
'ecg-id-database-1.0.0/Person_60/rec_1',
'ecg-id-database-1.0.0/Person_60/rec_2',
'ecg-id-database-1.0.0/Person_60/rec_3',
'ecg-id-database-1.0.0/Person_61/rec_1',
'ecg-id-database-1.0.0/Person_61/rec_2',
'ecg-id-database-1.0.0/Person_61/rec_3',
'ecg-id-database-1.0.0/Person_61/rec_4',
'ecg-id-database-1.0.0/Person_62/rec_1',
'ecg-id-database-1.0.0/Person_62/rec_2',
'ecg-id-database-1.0.0/Person_62/rec_3',
'ecg-id-database-1.0.0/Person_63/rec_1',
'ecg-id-database-1.0.0/Person_63/rec_2',
'ecg-id-database-1.0.0/Person_63/rec_3',
'ecg-id-database-1.0.0/Person_63/rec_4',
'ecg-id-database-1.0.0/Person_63/rec_5',
'ecg-id-database-1.0.0/Person_63/rec_6',
'ecg-id-database-1.0.0/Person_64/rec_1',
'ecg-id-database-1.0.0/Person_64/rec_2',
'ecg-id-database-1.0.0/Person_64/rec_3',
'ecg-id-database-1.0.0/Person_65/rec_1',
'ecg-id-database-1.0.0/Person_65/rec_2',
'ecg-id-database-1.0.0/Person_66/rec_1',
'ecg-id-database-1.0.0/Person_66/rec_2',
'ecg-id-database-1.0.0/Person_67/rec_1',
'ecg-id-database-1.0.0/Person_67/rec_2',
'ecg-id-database-1.0.0/Person_67/rec_3',
'ecg-id-database-1.0.0/Person_68/rec_1',
'ecg-id-database-1.0.0/Person_68/rec_2',
'ecg-id-database-1.0.0/Person_69/rec_1',
'ecg-id-database-1.0.0/Person_69/rec_2',
'ecg-id-database-1.0.0/Person_70/rec_1',
'ecg-id-database-1.0.0/Person_70/rec_2',
'ecg-id-database-1.0.0/Person_70/rec_3',
'ecg-id-database-1.0.0/Person_71/rec_1',
'ecg-id-database-1.0.0/Person_71/rec_2',
'ecg-id-database-1.0.0/Person_71/rec_3',
'ecg-id-database-1.0.0/Person_71/rec_4',
'ecg-id-database-1.0.0/Person_71/rec_5',
'ecg-id-database-1.0.0/Person_72/rec_1',
'ecg-id-database-1.0.0/Person_72/rec_2',
'ecg-id-database-1.0.0/Person_72/rec_3',
'ecg-id-database-1.0.0/Person_72/rec_4',
'ecg-id-database-1.0.0/Person_72/rec_5',
'ecg-id-database-1.0.0/Person_72/rec_6',
'ecg-id-database-1.0.0/Person_72/rec_7',
'ecg-id-database-1.0.0/Person_72/rec_8',
'ecg-id-database-1.0.0/Person_73/rec_1',
'ecg-id-database-1.0.0/Person_73/rec_2',
'ecg-id-database-1.0.0/Person_74/rec_1',
'ecg-id-database-1.0.0/Person_75/rec_1',
'ecg-id-database-1.0.0/Person_75/rec_2',
'ecg-id-database-1.0.0/Person_75/rec_3',
'ecg-id-database-1.0.0/Person_76/rec_1',
'ecg-id-database-1.0.0/Person_76/rec_2',
'ecg-id-database-1.0.0/Person_76/rec_3',
'ecg-id-database-1.0.0/Person_77/rec_1',
'ecg-id-database-1.0.0/Person_77/rec_2',
'ecg-id-database-1.0.0/Person_77/rec_3',
'ecg-id-database-1.0.0/Person_78/rec_1',
'ecg-id-database-1.0.0/Person_78/rec_2',
'ecg-id-database-1.0.0/Person_79/rec_1',
'ecg-id-database-1.0.0/Person_79/rec_2',
'ecg-id-database-1.0.0/Person_80/rec_1',
'ecg-id-database-1.0.0/Person_80/rec_2',
'ecg-id-database-1.0.0/Person_81/rec_1',
'ecg-id-database-1.0.0/Person_81/rec_2',
'ecg-id-database-1.0.0/Person_82/rec_1',
'ecg-id-database-1.0.0/Person_82/rec_2',
'ecg-id-database-1.0.0/Person_83/rec_1',
'ecg-id-database-1.0.0/Person_83/rec_2',
'ecg-id-database-1.0.0/Person_84/rec_1',
'ecg-id-database-1.0.0/Person_84/rec_2',
'ecg-id-database-1.0.0/Person_85/rec_1',
'ecg-id-database-1.0.0/Person_85/rec_2',
'ecg-id-database-1.0.0/Person_85/rec_3',
'ecg-id-database-1.0.0/Person_86/rec_1',
'ecg-id-database-1.0.0/Person_86/rec_2',
'ecg-id-database-1.0.0/Person_87/rec_1',
'ecg-id-database-1.0.0/Person_87/rec_2',
'ecg-id-database-1.0.0/Person_88/rec_1',
'ecg-id-database-1.0.0/Person_88/rec_2',
'ecg-id-database-1.0.0/Person_88/rec_3',
'ecg-id-database-1.0.0/Person_89/rec_1',
'ecg-id-database-1.0.0/Person_89/rec_2',
'ecg-id-database-1.0.0/Person_90/rec_1',
'ecg-id-database-1.0.0/Person_90/rec_2',
]

records = train_records

# Load ECG signal
def load_ecg_signal(file_path):
    try:
        record = wfdb.rdrecord(file_path)
        signal = record.p_signal[:, 0]  # Single-channel ECG
        return signal
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")

# Convert ECG to spectrogram
def ecg_to_spectrogram(signal, fs=256, nperseg=64):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
    return np.log(Sxx + 1e-6)  # Log scale

# Preprocess data
def preprocess_data(records, target_size=(224, 224)):
    spectrograms = []
    labels = []

    for record in records:
        signal = load_ecg_signal(record)
        spectrogram_data = ecg_to_spectrogram(signal)
        
        # Resize spectrogram to target size
        spectrogram_resized = tf.image.resize(
            np.expand_dims(spectrogram_data, axis=-1),  # Add channel dim
            target_size
        ).numpy()
        
        # Repeat to 3 channels (RGB)
        spectrogram_rgb = np.repeat(spectrogram_resized, 3, axis=-1)
        
        spectrograms.append(spectrogram_rgb)
        labels.append(1 if "Person_08" in record else 0)  # Binary label for Person_08
    
    return np.array(spectrograms), np.array(labels)

# Create MobileNetV1 + GRU model
def create_mobilenet_gru_model(input_shape=(224, 224, 3)):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Reshape((1, 1024)),  # Explicitly define time steps
        layers.GRU(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),  # Add dropout to prevent overfitting
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Preprocess training data
X_train, y_train = preprocess_data(train_records)
X_train = (X_train - np.mean(X_train)) / np.std(X_train)  # Standardize using mean/std
print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")

# Create and compile model
model = create_mobilenet_gru_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train with callbacks
history = model.fit(
    X_train, 
    y_train, 
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
)

# Test records
test_records = [
    'ecg-id-database-1.0.0/Person_01/rec_1',
    'ecg-id-database-1.0.0/Person_08/rec_1',
    'ecg-id-database-1.0.0/Person_01/rec_3',
]

# Evaluate on test data
X_test, y_test = preprocess_data(test_records)
X_test = (X_test - np.mean(X_train)) / np.std(X_train)  # Use training stats for normalization

for i, record in enumerate(test_records):
    prediction = model.predict(X_test[i:i+1])[0][0]
    print(f"Record: {record}")
    print(f"Predicted Probability: {prediction:.4f}")
    print("Status: Authenticated ✅" if prediction >= 0.5 else "Status: Not Authenticated ❌")
    print()

# Overall test evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")