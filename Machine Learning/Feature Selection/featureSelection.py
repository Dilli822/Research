import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# Dictionary of dataset URLs with settings for each
datasets = {
    "1": {
        "name": "Wine Quality",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "delimiter": ";",
        "header": 0,
        "target_column": -1,
        "description": "Quality distribution of wine samples"
    },
    "2": {
        "name": "Breast Cancer Wisconsin",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "delimiter": ",",
        "header": None,
        "target_column": 1,
        "description": "Diagnosis distribution (M=malignant, B=benign)"
    },
    "3": {
        "name": "Pima Indians Diabetes",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "delimiter": ",",
        "header": None,
        "target_column": -1,
        "description": "Distribution of diabetes cases"
    },
    "4": {
        "name": "Heart Disease",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "delimiter": ",",
        "header": None,
        "target_column": -1,
        "description": "Heart disease presence (0 = no, 1 = yes)"
    },
    "5": {
        "name": "Iris",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "delimiter": ",",
        "header": None,
        "target_column": -1,
        "description": "Iris species distribution"
    },
    "6": {
        "name": "Titanic",
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "delimiter": ",",
        "header": 0,
        "target_column": -1,
        "description": "Survival distribution (1=survived, 0=not survived)"
    },
    "7": {
        "name": "Adult",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "delimiter": ",",
        "header": None,
        "target_column": -1,
        "description": "Income distribution (<=50K or >50K)"
    },
    "8": {
        "name": "Abalone",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        "delimiter": ",",
        "header": None,
        "target_column": -1,
        "description": "Age distribution of abalones (based on rings)"
    }
}

# Loop until the user decides to exit
while True:
    # Display options to the user
    print("\nSelect a dataset by entering the corresponding number (or enter 0 to exit):")
    for key, info in datasets.items():
        print(f"{key}: {info['name']}")
    print("0: Exit")

    # Get user input
    choice = input("Enter the number of the dataset you want to load: ")

    # Exit condition
    if choice == "0":
        print("Exiting the program.")
        break

    # Check if choice is valid
    if choice in datasets:
        dataset = datasets[choice]
        print(f"\nLoading dataset: {dataset['name']}")

        # Load the dataset
        df = pd.read_csv(dataset["url"], delimiter=dataset["delimiter"], header=dataset["header"])

        # Separate features and target
        X = df.iloc[:, :-1] if dataset["target_column"] == -1 else df.drop(dataset["target_column"], axis=1)
        y = df.iloc[:, dataset["target_column"]]

        # Display first few rows of X and y
        print("\nFeature Table (X):")
        print(tabulate(X.head(), headers=X.columns, tablefmt="grid"))
        print("\nTarget Table (y):")
        print(tabulate(y.head().to_frame(), headers=["target"], tablefmt="grid"))

        # Plot histograms for each feature column
        X.hist(bins=5, figsize=(15, 10), color='skyblue', grid=False)
        plt.suptitle(f'Feature Distributions in {dataset["name"]}')
        plt.show()

        # Plot the distribution of the target variable
        plt.figure(figsize=(8, 5))
        y.value_counts().sort_index().plot(kind='bar', color='salmon')
        plt.title(f'{dataset["name"]} - {dataset["description"]}')
        plt.xlabel('Target Classes')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)
        plt.show()
    else:
        print("Invalid selection. Please enter a valid number.")
