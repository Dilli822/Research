import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Reshape, Lambda, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Recall, Precision, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

# --- Step 1: Prepare Dummy Data (Replace with real data) ---
# Create 150 spectrogram samples of size 128x128
X = np.random.rand(950, 128, 128)

# Create 150 corresponding binary labels (e.g., alternating or random)
y = np.random.randint(0, 2, size=(950,))

# Add a channel dimension: (N, 128, 128, 1)
X = X[..., np.newaxis]

# --- Step 2: Build MobileNetV2 + GRU Hybrid Model ---

def expand_to_rgb(x):
    return tf.repeat(x, repeats=3, axis=-1)

input_shape = (128, 128, 1)
input_layer = Input(shape=input_shape)

# Convert grayscale to RGB
rgb_input = Lambda(expand_to_rgb)(input_layer)

# MobileNetV2 as feature extractor
mobilenet_base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
mobilenet_base.trainable = False

features = mobilenet_base(rgb_input)
pooled = GlobalAveragePooling2D()(features)         # shape: (batch_size, features)
reshaped = Reshape((1, -1))(pooled)                 # shape: (batch_size, 1, features)

# GRU Layer
gru_out = GRU(64, return_sequences=False)(reshaped)

# Final Dense output (binary classification)
output = Dense(1, activation='sigmoid')(gru_out)

# Build Model
model = Model(inputs=input_layer, outputs=output)

# Compile model with additional metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy', 
        Precision(),
        Recall(),
        AUC(),
    ]
)

model.summary()

# --- Step 3: Train-Test Split and Fit ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=2
)

# --- Step 4: Plot Training History ---
# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --- Step 5: ROC Curve Plotting ---
# Get predicted probabilities
y_pred_prob = model.predict(X_val)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# --- Step 6: Print all validation metrics ---
print("\nValidation Metrics at the end of training:")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Validation Precision: {history.history['val_precision'][-1]:.4f}")
print(f"Final Validation Recall: {history.history['val_recall'][-1]:.4f}")
print(f"Final Validation AUC: {history.history['val_auc'][-1]:.4f}")
