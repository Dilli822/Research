import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Load the dataset into a DataFrame
df = pd.read_csv(data_url, names=columns)

# Separate features and target variable
X = df.drop('Outcome', axis=1)  
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# Function to make predictions with user inputs
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Return the prediction
    return prediction[0]  # 0 for No Diabetes, 1 for Diabetes

# Example input
predicted_outcome = predict_diabetes(1, 85, 66, 29, 0, 26.6, 0.351, 31)  # Example Input values
print("Predicted Outcome (0 = No Diabetes, 1 = Diabetes):", predicted_outcome)



