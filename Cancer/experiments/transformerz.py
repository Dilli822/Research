import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  # Change to float32 for BCELoss

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Specify columns to use (all columns except target)
X_n_columns = list(range(X_tensor.shape[1]))  # Use all features except target column
X_n_tensor = X_tensor[:, X_n_columns]  # Select all columns

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_encoder_layers=3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, 1)  # Binary classification

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, input_dim) -> (batch_size, d_model)
        x = x.unsqueeze(1)  # Add sequence length dimension: (batch_size, d_model) -> (batch_size, 1, d_model)
        x = self.transformer_encoder(x)  # Shape: (batch_size, 1, d_model)
        x = x.squeeze(1)  # Remove sequence length dimension
        x = self.fc(x)  # Shape: (batch_size, d_model) -> (batch_size, 1)
        return torch.sigmoid(x)

# Initialize model, loss, and optimizer
input_dim = len(X_n_columns)
model = TransformerModel(input_dim)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch[:, X_n_columns]  # Select only specified columns
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extract input features function
def get_input_features(model, X):
    model.eval()
    with torch.no_grad():
        # Pass input data through the model's embedding layer
        features = model.embedding(X)
        return features.cpu().numpy()

# Function to compute and display feature importance
def get_most_important_features(model, feature_names, top_n=15):
    # Extract the embedding layer weights
    embedding_weights = model.embedding.weight.detach().cpu().numpy()
    feature_importances = abs(embedding_weights).sum(axis=1)  # Sum of absolute weights per feature
    
    # Pair feature importances with their names
    feature_importance_pairs = list(zip(feature_names, feature_importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by importance
    
    # Get top N important features
    print(f"\nTop {top_n} Important Features:")
    for idx, (feature_name, importance) in enumerate(feature_importance_pairs[:top_n]):
        print(f"{idx + 1}. {feature_name}: Importance = {importance:.4f}")
    
    return feature_importance_pairs[:top_n]

# Specify feature names for the selected columns
selected_feature_names = data.feature_names

# Get input features for a sample input
sample_input = X_tensor[0].unsqueeze(0)[:, X_n_columns]  # Select only the specified columns
input_features = get_input_features(model, sample_input)

# Print the extracted input features
print("Extracted Input Features:")
for idx, feature_value in enumerate(input_features[0]):
    print(f"Feature {idx}: {feature_value:.4f}")

# Display the most important features
top_features = get_most_important_features(model, selected_feature_names, top_n=15)


# Extract features for the entire dataset
extracted_features = get_input_features(model, X_tensor[:, X_n_columns])

# Split extracted features into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(extracted_features, y, test_size=0.3, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# Initialize classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42,probability=True),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(train_features, train_labels)
    
    # Predict on test data
    predictions = clf.predict(test_features)
    
    # Evaluate performance
    acc = accuracy_score(test_labels, predictions)
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(test_labels, predictions))


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Collect results for visualization
results = []

for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(train_features, train_labels)
    
    # Predict on test data
    predictions = clf.predict(test_features)
    
    # Evaluate performance
    acc = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)  # Convert report to dictionary
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    # Store the results
    results.append({
        "Classifier": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_df.columns))))

# Save the table as a PNG file
plt.savefig("classifier_results.png", bbox_inches='tight', dpi=300)
plt.show()
