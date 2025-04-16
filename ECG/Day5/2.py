import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Resize images to 96x96 and normalize
x_train = tf.image.resize(x_train[..., tf.newaxis], (96, 96)) / 255.0
x_test = tf.image.resize(x_test[..., tf.newaxis], (96, 96)) / 255.0

# Convert grayscale to RGB
x_train_rgb = tf.image.grayscale_to_rgb(x_train)
x_test_rgb = tf.image.grayscale_to_rgb(x_test)

# Prepare tf.data datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_rgb, y_train)).batch(16).shuffle(1000)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_rgb, y_test)).batch(16)


base_model = tf.keras.applications.MobileNet(
    input_shape=(96, 96, 3),
    include_top=False,
    weights=None
)

# Optional: Freeze base layers
base_model.trainable = False

# Add classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_dataset, epochs=1, validation_data=test_dataset)

model.summary()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")



# Take a few samples from the test dataset
for images, labels in test_dataset.take(1):  # take first batch
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Show 5 images with predicted and actual labels
    for i in range(5):
        plt.imshow(images[i].numpy().astype("float32"))
        plt.title(f"Predicted: {predicted_labels[i]}, Actual: {labels[i].numpy()}")
        plt.axis('off')
        plt.show()
        


def predict_from_url(image_url, model):
    try:
        # Load image from URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Resize and normalize
        img = img.resize((96, 96))
        img_array = np.array(img) / 255.0
        img_tensor = tf.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(img_tensor)
        predicted_label = np.argmax(predictions)

        # Show image and prediction
        plt.imshow(img)
        plt.title(f"Predicted Digit: {predicted_label}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("⚠️ Error:", e)

# Continuous input loop
while True:
    image_url = input("🔗 Enter image URL (or type 'exit' to quit): ")
    if image_url.lower() == 'exit':
        print("👋 Exiting prediction loop.")
        break
    predict_from_url(image_url, model)
