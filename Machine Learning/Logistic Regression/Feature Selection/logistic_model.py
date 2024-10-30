# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Sample dataset: Let's create a simple dataset
# # Features: Hours studied, Hours slept
# # Labels: 0 = Fail, 1 = Pass
# data = {
#     'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'Hours_Slept':   [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
#     'Result':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Split the data into features and target variable
# X = df[['Hours_Studied', 'Hours_Slept']]
# y = df['Result']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Create a Logistic Regression model
# model = LogisticRegression()

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# # Display results
# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", class_report)


# # -- matter --> molecules --> atom --> electron ---> nuclues --> neutron+proton --> quark --> string

import pandas as pd
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters (weights and bias)
def initialize_params(dim):
    W = np.zeros((dim, 1))
    b = 0
    return W, b

# Forward and backward propagation
def propagate(W, b, X, Y):
    m = X.shape[1]  # Number of samples
    
    # Forward propagation (Z = WX + b, A = sigmoid(Z))
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    
    # Compute cost (binary cross-entropy loss)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    # Backward propagation (compute gradients)
    dW = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    
    # Return gradients and cost
    grads = {"dW": dW, "db": db}
    return grads, cost

# Gradient descent optimization
def optimize(W, b, X, Y, learning_rate, num_iterations):
    costs = []
    
    for i in range(num_iterations):
        # Get gradients and cost using forward and backward propagation
        grads, cost = propagate(W, b, X, Y)
        
        # Retrieve gradients
        dW = grads["dW"]
        db = grads["db"]
        
        # Update weights and bias
        W = W - learning_rate * dW
        b = b - learning_rate * db
        
        # Record the cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Iteration {i}, Cost: {cost}")
    
    # Return the final parameters and the list of costs
    params = {"W": W, "b": b}
    return params, costs

# Prediction using learned parameters
def predict(W, b, X):
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    predictions = A > 0.5  # If A > 0.5, classify as 1 (positive), else 0
    return predictions.astype(int)

# Logistic Regression Model
def logistic_regression_model(X_train, Y_train, learning_rate=0.01, num_iterations=1000):
    # Initialize parameters
    dim = X_train.shape[0]  # Number of features
    W, b = initialize_params(dim)
    
    # Optimize (train the model)
    params, costs = optimize(W, b, X_train, Y_train, learning_rate, num_iterations)
    
    # Retrieve final parameters
    W = params["W"]
    b = params["b"]
    
    # Return the model parameters
    return W, b, costs

# Example usage with Pima Indians dataset

# Load dataset
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
features = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(data_url, names=features)

# Prepare the data
X = df.iloc[:, 0:8].values.T  # Features (transpose to match shape for matrix multiplication)
Y = df.iloc[:, 8].values.reshape(1, -1)  # Target (reshape to a row vector)

# Standardize the dataset (optional but recommended for logistic regression)
X = (X - np.mean(X, axis=1).reshape(-1, 1)) / np.std(X, axis=1).reshape(-1, 1)

# Train the logistic regression model
W, b, costs = logistic_regression_model(X, Y, learning_rate=0.01, num_iterations=2000)

# Predict on training data
Y_pred_train = predict(W, b, X)
accuracy = np.mean(Y_pred_train == Y) * 100
print(f"Training Accuracy: {accuracy}%")
