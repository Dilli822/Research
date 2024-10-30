# Import necessary libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the wine quality dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
column_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 
                'density', 'pH', 'sulphates', 'alcohol', 'quality']

# Load the dataset
dataset = pd.read_csv(url, sep=';')  # Use ';' as the separator

# Display basic information
print(dataset.info())
print(dataset.head())

# Preprocessing: Encode target variable
dataset['quality'] = dataset['quality'].map(lambda x: 1 if x >= 7 else 0)  # Binary classification

# Split into train and test datasets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=1)

# Split features and labels
x_train, y_train = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1]
x_test, y_test = test_dataset.iloc[:, :-1], test_dataset.iloc[:, -1]

# Convert data to tensors
x_train, y_train = tf.convert_to_tensor(x_train.values, dtype=tf.float32), tf.convert_to_tensor(y_train.values, dtype=tf.float32)
x_test, y_test = tf.convert_to_tensor(x_test.values, dtype=tf.float32), tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Normalize the dataset
class Normalize(tf.Module):
    def __init__(self, x):
        self.mean = tf.Variable(tf.math.reduce_mean(x, axis=0))
        self.std = tf.Variable(tf.math.reduce_std(x, axis=0))
        
    def norm(self, x):
        return (x - self.mean) / self.std

norm_x = Normalize(x_train)
x_train_norm, x_test_norm = norm_x.norm(x_train), norm_x.norm(x_test)

# Logistic Regression Model
class LogisticRegression(tf.Module):
    def __init__(self):
        self.built = False
        
    def __call__(self, x, train=True):
        if not self.built:
            rand_w = tf.random.uniform(shape=[x.shape[-1], 1], seed=22)
            rand_b = tf.random.uniform(shape=[], seed=22)
            self.w = tf.Variable(rand_w)
            self.b = tf.Variable(rand_b)
            self.built = True
            
        z = tf.add(tf.matmul(x, self.w), self.b)
        z = tf.squeeze(z, axis=1)
        if train:
            return z
        return tf.sigmoid(z)

# Log loss function
def log_loss(y_pred, y):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(ce)

# Accuracy function
def accuracy(y_pred, y):
    y_pred = tf.math.sigmoid(y_pred)
    y_pred_class = predict_class(y_pred)
    check_equal = tf.cast(y_pred_class == y, tf.float32)
    return tf.reduce_mean(check_equal)

def predict_class(y_pred, thresh=0.5):
    return tf.cast(y_pred > thresh, tf.float32)

# Training loop
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train)).shuffle(buffer_size=x_train.shape[0]).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test)).batch(batch_size)

epochs = 200
learning_rate = 0.01
train_losses, test_losses = [], []
train_accs, test_accs = [], []

log_reg = LogisticRegression()

for epoch in range(epochs):
    batch_losses_train, batch_accs_train = [], []
    batch_losses_test, batch_accs_test = [], []
    
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            y_pred_batch = log_reg(x_batch)
            batch_loss = log_loss(y_pred_batch, y_batch)
            batch_acc = accuracy(y_pred_batch, y_batch)
        grads = tape.gradient(batch_loss, log_reg.variables)
        for g, v in zip(grads, log_reg.variables):
            v.assign_sub(learning_rate * g)
        batch_losses_train.append(batch_loss)
        batch_accs_train.append(batch_acc)

    for x_batch, y_batch in test_dataset:
        y_pred_batch = log_reg(x_batch)
        batch_loss = log_loss(y_pred_batch, y_batch)
        batch_acc = accuracy(y_pred_batch, y_batch)
        batch_losses_test.append(batch_loss)
        batch_accs_test.append(batch_acc)
    
    train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
    test_loss, test_acc = tf.reduce_mean(batch_losses_test), tf.reduce_mean(batch_accs_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Training log loss: {train_loss:.3f}")

# Plot loss and accuracy
plt.plot(range(epochs), train_losses, label="Training loss")
plt.plot(range(epochs), test_losses, label="Testing loss")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.legend()
plt.title("Log loss vs training iterations")
plt.show()

plt.plot(range(epochs), train_accs, label="Training accuracy")
plt.plot(range(epochs), test_accs, label="Testing accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs training iterations")
plt.show()

# Print final results
train_accuracy = train_accs[-1]
train_loss = train_losses[-1]
test_accuracy = test_accs[-1]
test_loss = test_losses[-1]

print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}, Testing Loss: {test_loss:.4f}")

# Check for overfitting, underfitting, or normal fitting
if train_accuracy > test_accuracy and train_loss < test_loss:
    print("Model is overfitting.")
elif train_accuracy < 0.7:  # Arbitrary threshold for underfitting
    print("Model is underfitting.")
else:
    print("Model is fitting normally.")
