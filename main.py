import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve


# Ex 1
X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42,
)

plt.figure()
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            color="blue", label="Normal")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            color="red", label="Outliers")
plt.title("Ex 1 - Training data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

plt.show()
