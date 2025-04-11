import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Reshape, Lambda, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# --- Step 1: Prepare Dummy Data ---
X = np.random.rand(750, 128, 128, 1) / 255.0  # Normalize
y = np.random.randint(0, 2, size=(750,))

# --- Step 2: Build Model ---
def expand_to_rgb(x):
    return tf.repeat(x, repeats=3, axis=-1)

input_shape = (128, 128, 1)
input_layer = Input(shape=input_shape)
rgb_input = Lambda(expand_to_rgb)(input_layer)

mobilenet_base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
mobilenet_base.trainable = False

features = mobilenet_base(rgb_input)
pooled = GlobalAveragePooling2D()(features)
reshaped = Reshape((1, -1))(pooled)
gru_out = GRU(64, return_sequences=False)(reshaped)
dropout = Dropout(0.3)(gru_out)
output = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=input_layer, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

# --- Step 3: Train-Test Split and Augmentation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# --- Step 4: Train with Early Stopping ---
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=[early_stopping]
)

# --- Step 5: Plotting and Evaluation ---
# Print metrics
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Validation Precision: {history.history.get('val_precision', history.history.get('val_precision_1'))[-1]:.4f}")
print(f"Final Validation Recall: {history.history.get('val_recall', history.history.get('val_recall_1'))[-1]:.4f}")
print(f"Final Validation AUC: {history.history.get('val_auc', history.history.get('val_auc_1'))[-1]:.4f}")

# ROC curve plotting
y_pred_prob = model.predict(X_val)
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
