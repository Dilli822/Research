import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None

    def load_dataset(self, file_path):
        """
        Load dataset from file path
        """
        try:
            # Try to read with different encodings
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')

            print("\nDataset loaded successfully!")
            self.explore_data(df)
            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def explore_data(self, df):
        """
        Explore and display dataset information
        """
        print("\n=== Dataset Overview ===")
        print(f"Shape: {df.shape} (rows, columns)")

        print("\n=== Column Information ===")
        for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
            unique_vals = df[col].nunique()
            print(f"{i}. {col}")
            print(f"   Type: {dtype}")
            print(f"   Unique values: {unique_vals}")
            if unique_vals <= 10:
                print(f"   Values: {sorted(df[col].unique())}")
            print()

        print("\n=== Missing Values ===")
        missing = df.isnull().sum()
        if missing.any():
            print(missing[missing > 0])
        else:
            print("No missing values found")

    def prepare_data(self, df, target_column):
        """
        Prepare data for modeling
        """
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found!")
            return None, None

        try:
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Store feature names
            self.feature_names = X.columns.tolist()

            # Handle categorical variables in features
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # Handle categorical target
            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)

            # Handle missing values
            X = X.fillna(X.mean())

            return X, y

        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None

    def train_model(self, X, y):
        """
        Train the logistic regression model
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_prob = self.model.predict_proba(X_test_scaled)

            # Evaluate model
            self.evaluate_model(y_test, y_pred)

            # Plot results
            self.plot_results(X, y_test, y_pred, y_pred_prob)

            return self.model

        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def evaluate_model(self, y_test, y_pred):
        """
        Evaluate and display model performance
        """
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def plot_results(self, X, y_test, y_pred, y_pred_prob):
        """
        Create and display visualization plots
        """
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # ROC Curve for binary classification
        if len(np.unique(y_test)) == 2:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.show()

        # Feature Importance
        plt.figure(figsize=(10, 6))
        importance = abs(self.model.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()
        plt.show()

    def predict_new(self, new_data):
        """
        Predict with new data points
        """
        try:
            if self.model is None or self.scaler is None:
                print("Model has not been trained or scaler is not available.")
                return None
            
            # Ensure new data has the same columns as the original features
            if len(new_data.columns) != len(self.feature_names):
                print("New data does not match the feature format of the training data.")
                return None
            
            # Handle categorical variables if necessary
            for col in new_data.columns:
                if new_data[col].dtype == 'object':
                    le = LabelEncoder()
                    new_data[col] = le.fit_transform(new_data[col].astype(str))
            
            # Scale new data
            new_data_scaled = self.scaler.transform(new_data)
            
            # Make predictions
            predictions = self.model.predict(new_data_scaled)
            prediction_probs = self.model.predict_proba(new_data_scaled)
            
            return predictions, prediction_probs

        except Exception as e:
            print(f"Error predicting new data: {e}")
            return None

def main():
    """
    Main function to run the analysis
    """
    analyzer = DatasetAnalyzer()

    print("Welcome to Logistic Regression Analysis!")
    print("======================================")

    # Get dataset path
    file_path = input("\nEnter the path to your dataset (CSV file): ")

    # Load and explore dataset
    df = analyzer.load_dataset(file_path)
    if df is None:
        return

    # Get target column
    target_column = input("\nEnter the name of your target column: ")

    # Prepare data
    X, y = analyzer.prepare_data(df, target_column)
    if X is None or y is None:
        return

    # Train and evaluate model
    print("\nTraining model...")
    model = analyzer.train_model(X, y)

    if model is not None:
        print("\nModel training completed successfully!")

        # Testing new data
        test = input("\nWould you like to test the model with new data? (yes/no): ")
        if test.lower() == 'yes':
            new_data = pd.DataFrame({
                # Replace with actual data points
                'Pregnancies': [6],
                'Glucose': [148],
                'BloodPressure': [72],
                'SkinThickness': [35],
                "Insulin": [0],
                "BMI": [33.6],
                "DiabetesPedigreeFunction": [0.627],
                "Age": [50],
            })
            predictions, prediction_probs = analyzer.predict_new(new_data)

            if predictions is not None:
                print("\nNew Data Predictions:")
                print(f"Predictions: {predictions}")
                print(f"Prediction Probabilities: {prediction_probs}")

if __name__ == "__main__":
    main()
