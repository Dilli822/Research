import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Step 1: Prepare Data Sample sentences
sentences = [
    "I love programming", 
    "Python is awesome", 
    "I enjoy learning machine learning", 
    "Deep learning is exciting", 
    "I hate bugs", 
    "Debugging is frustrating", 
    "I love solving problems", 
    "Python makes life easier"
]

# Step 2: Tokenize and Preprocess the Text
# We need to tokenize the text (convert words into numbers) and pad the sequences to ensure they are of equal length:
# Labels: 1 for positive, 0 for negative
labels = [1, 1, 1, 1, 0, 0, 1, 1]


# Initialize the Tokenizer
tokenizer = Tokenizer(num_words=10000)  # We limit the vocabulary size to 10,000 words
tokenizer.fit_on_texts(sentences)

# Convert sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to ensure they all have the same length
padded_sequences = pad_sequences(sequences, padding='post', maxlen=10)  # maxlen ensures uniform length


# Step 3: Define the GRU Model
# Now, we’ll build a simple GRU-based neural network for text classification: Define the model architecture
model = Sequential()

# Embedding layer to convert integer tokens to dense vectors
model.add(Embedding(input_dim=10000, output_dim=128, input_length=10))  # 10 is the maxlen we defined earlier

# GRU layer to capture sequential dependencies in the text
model.add(GRU(64, return_sequences=False))  # 64 is the number of GRU units

# Dropout layer for regularization to avoid overfitting
model.add(Dropout(0.1))

# Output layer: Sigmoid activation for binary classification (positive or negative)
model.add(Dense(1, activation='sigmoid'))

# Step 4: Compile the Model
# Next, we’ll compile the model using the Adam optimizer and binary cross-entropy loss, as we are doing binary classification:
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
# Now, we can train the model on our data. We'll use the padded sequences and labels for training:
model.fit(padded_sequences, np.array(labels), epochs=35, batch_size=2)

# Step 8: Evaluate the Model
# Once the model is trained, we can evaluate it on new data:
    
# Example test sentence
test_sentences = ["I love coding", "I don't like errors!"]

# Preprocess the test data
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=10)

# Predict using the trained model
predictions = model.predict(test_padded)

# Print the predictions (0 for negative, 1 for positive)
for sentence, prediction in zip(test_sentences, predictions):
    print(f"Sentence: '{sentence}' -> Prediction: {'Positive' if prediction > 0.5 else 'Negative'}")


"""
Explanation:
Embedding Layer: This layer converts each word into a dense vector. We’ve used a vector size of 128 for the embedding.

GRU Layer: The GRU layer processes sequential data, capturing dependencies between words in the sentence. We set return_sequences=False to output only the final hidden state.

Dropout Layer: Helps regularize the model to avoid overfitting.

Dense Layer: The final output layer uses the sigmoid activation function to output a value between 0 and 1, representing the probability of the positive class.

Training: We used the binary cross-entropy loss function because it's a binary classification task, and accuracy as the evaluation metric.
"""