
import numpy as np
from sklearn.model_selection import train_test_split

# Sample array
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



# The splits will vary each time you run the code, leading to different training and testing sets.
X_train_1, X_test_1 = train_test_split(data, test_size=0.3)
print("Split 1 without random state:")
print("X_train:", X_train_1)
print("X_test:", X_test_1)


X_train_2, X_test_2 = train_test_split(data, test_size=0.3)
print("\nSplit 2 without random state:")
print("X_train:", X_train_2)
print("X_test:", X_test_2)

# You will get the same split every time, which is crucial for reproducibility in machine learning experiments.

print("""
This makes it easier to compare results and debug your model, as you can ensure that any changes in performance are
 due to the model or parameters you’re testing, not random variations in the data split.
""")
X_train_3, X_test_3 = train_test_split(data, test_size=0.3, random_state=42)
print("\nSplit 3 (random_state=42) You will get the same split every time, which is crucial for reproducibility in machine learning experiments.:")
print("X_train:", X_train_3)
print("X_test:", X_test_3)

X_train_4, X_test_4 = train_test_split(data, test_size=0.3, random_state=42)
print("\nSplit 4 (random_state=42):")
print("X_train:", X_train_4)
print("X_test:", X_test_4)
