import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Load the pre-trained MobileNetV1 model manually
model = tf.keras.applications.MobileNet(weights='imagenet')

# Image preprocessing pipeline for MobileNetV1
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to 224x224
    img_array = np.array(img)  # Convert image to numpy array
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)  # Preprocess for MobileNet
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to load an image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Image URL (you can replace this with any image URL)
img_url = 'https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500'  # Replace with your image URL

# Load the image from the URL
img = load_image_from_url(img_url)

# Preprocess the image for MobileNetV1
img_preprocessed = preprocess_image(img)

# Predict with the model
predictions = model.predict(img_preprocessed)

# Decode the predictions
decoded_predictions = tf.keras.applications.mobilenet.decode_predictions(predictions, top=3)[0]

# Print the top 3 predicted classes
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}. {label}: {score:.2f}")

# Visualize the image with its prediction
plt.imshow(img)
plt.title(f"Predicted: {decoded_predictions[0][1]} ({decoded_predictions[0][2]:.2f})")
plt.show()
