# Import necessary libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
from sklearn.metrics import classification_report


# Load the Adult Income dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Load the dataset
dataset = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

# Display basic information
print(dataset.info())
print(dataset.head())

# Preprocessing: Drop missing values and encode categorical features
dataset.dropna(inplace=True)
dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1})

# One-hot encode categorical features
dataset = pd.get_dummies(dataset, columns=['workclass', 'education', 'marital-status', 
                                            'occupation', 'relationship', 'race', 'sex', 
                                            'native-country'], drop_first=True)


# same as other code
