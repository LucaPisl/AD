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


# Ex 2
# 0 means normal, 1 means anomaly
def print_metrics(y, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2
    print("TN:", tn, "TP:", tp, "FN:", fn, "FP:", fp)
    print("Balanced accuracy:", round(balanced_accuracy, 4))


for contamination in [0.1, 0.05, 0.2]:
    model = KNN(contamination=contamination)
    model.fit(X_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    print("\nEx 2 - contamination:", contamination)
    print("Train:")
    print_metrics(y_train, train_prediction)
    print("Test:")
    print_metrics(y_test, test_prediction)

    # ROC needs the scores, not the 0/1 predictions
    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)
    train_fpr, train_tpr, _ = roc_curve(y_train, train_scores)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_scores)

    plt.figure()
    plt.plot(train_fpr, train_tpr, label="Train")
    plt.plot(test_fpr, test_tpr, label="Test")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Ex 2 - contamination " + str(contamination))
    plt.legend()

# More contamination lowers the threshold, so more points become anomalies.
# The ROC stays the same here because KNN scores don't depend on contamination.


# Ex 3
X, y = generate_data(n_train=1000, n_test=0, n_features=1,
                     contamination=0.1, train_only=True, random_state=42)

z_scores = np.abs((X[:, 0] - np.mean(X[:, 0])) / np.std(X[:, 0]))
# Take the largest 10% of scores
threshold = np.quantile(z_scores, 0.9)
prediction = (z_scores > threshold).astype(int)

print("\nEx 3")
print("Threshold:", threshold)
print_metrics(y, prediction)


# Ex 4
np.random.seed(42)
mu = np.array([2, 3])
sigma = np.array([[2, 1], [1, 2]])
L = np.linalg.cholesky(sigma)

x = np.random.normal(0, 1, (900, 2))
# Each sample is a row here, so Lx becomes x @ L.T
normal = x @ L.T + mu
outliers = np.random.uniform(-8, 12, (100, 2))
X = np.vstack((normal, outliers))
y = np.concatenate((np.zeros(900), np.ones(100)))

# Standardize each column, then combine the two scores into a distance
z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
z_scores = np.sqrt(np.sum(z ** 2, axis=1))
threshold = np.quantile(z_scores, 0.9)
prediction = (z_scores > threshold).astype(int)

print("\nEx 4")
print("Threshold:", threshold)
print_metrics(y, prediction)

plt.show()
