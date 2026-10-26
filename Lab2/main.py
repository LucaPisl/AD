import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score


# Ex 1
np.random.seed(42)
a, b = 2, 1
n_regular = 100
n_other = 5
top_k = 10
names = ["regular", "high var x", "high var y", "high var x and y"]
colors = ["blue", "green", "orange", "purple"]


def leverage_scores(X):
    # extra column of ones for the intercept
    A = np.column_stack((X, np.ones(len(X))))
    H = A @ np.linalg.inv(A.T @ A) @ A.T
    return np.diag(H)


def make_data_1d(mu, var):
    std = np.sqrt(var)
    low = np.sqrt(0.1)
    sizes = [n_regular, n_other, n_other, n_other]
    x_std = [1, std, 1, std]
    eps_mu = [0, 0, mu, mu]
    eps_std = [low, low, std, std]

    x = np.concatenate([np.random.normal(0, x_std[i], sizes[i]) for i in range(4)])
    eps = np.concatenate([np.random.normal(eps_mu[i], eps_std[i], sizes[i]) for i in range(4)])
    types = np.repeat(np.arange(4), sizes)
    return x, a * x + b + eps, types


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (mu, var) in zip(axes, [(0, 4), (1, 16), (2, 36)]):
    x, y, types = make_data_1d(mu, var)
    scores = leverage_scores(x)
    top = np.argsort(scores)[-top_k:]

    for t in range(4):
        ax.scatter(x[types == t], y[types == t], s=15, color=colors[t], label=names[t])
    ax.scatter(x[top], y[top], s=120, facecolors="none", edgecolors="red",
               linewidths=1.5, label="top leverage")
    ax.set_title("mu = " + str(mu) + ", var = " + str(var))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)

    print("\nEx 1 (1D) mu:", mu, "var:", var)
    print("Top", top_k, "leverage points by type:", np.bincount(types[top], minlength=4))

fig.suptitle("Ex 1 - leverage scores, 1D")

# Leverage only depends on x. The points far away on x (green and purple) get
# the highest scores, the orange ones are not picked more than the regular ones.
# Some regular points still show up in the top because top_k is fixed.


# Ex 1 - 2D case, y = a*x1 + b*x2 + c + eps
a2, b2, c2 = 2, -1, 1


def make_data_2d(mu, var):
    std = np.sqrt(var)
    low = np.sqrt(0.1)
    sizes = [n_regular, n_other, n_other, n_other]
    x_std = [1, std, 1, std]
    eps_mu = [0, 0, mu, mu]
    eps_std = [low, low, std, std]

    X = np.vstack([np.random.normal(0, x_std[i], (sizes[i], 2)) for i in range(4)])
    eps = np.concatenate([np.random.normal(eps_mu[i], eps_std[i], sizes[i]) for i in range(4)])
    types = np.repeat(np.arange(4), sizes)
    return X, a2 * X[:, 0] + b2 * X[:, 1] + c2 + eps, types


fig = plt.figure(figsize=(15, 5))
for i, (mu, var) in enumerate([(0, 4), (1, 16), (2, 36)]):
    X, y, types = make_data_2d(mu, var)
    scores = leverage_scores(X)
    top = np.argsort(scores)[-top_k:]

    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    for t in range(4):
        ax.scatter(X[types == t, 0], X[types == t, 1], y[types == t],
                   s=15, color=colors[t], label=names[t])
    ax.scatter(X[top, 0], X[top, 1], y[top], s=120, facecolors="none",
               edgecolors="red", linewidths=1.5, label="top leverage")
    ax.set_title("mu = " + str(mu) + ", var = " + str(var))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.legend(fontsize=7)

    print("\nEx 1 (2D) mu:", mu, "var:", var)
    print("Top", top_k, "leverage points by type:", np.bincount(types[top], minlength=4))

fig.suptitle("Ex 1 - leverage scores, 2D")

# Same thing in 2D: the high var on y points never end up in the top, and the
# more var we have on x1/x2 the more the top is made of green and purple points.


# Ex 2
X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_clusters=2,
    n_features=2,
    contamination=0.1,
    random_state=42,
)


def plot_labels(ax, X, labels, title):
    # 0 means normal, 1 means anomaly
    ax.scatter(X[labels == 0, 0], X[labels == 0, 1], s=15, color="blue", label="Inliers")
    ax.scatter(X[labels == 1, 0], X[labels == 1, 1], s=15, color="red", label="Outliers")
    ax.set_title(title)


for n_neighbors in [1, 5, 20, 50, 100]:
    model = KNN(contamination=0.1, n_neighbors=n_neighbors)
    model.fit(X_train)

    # labels_ leaves each point out of its own neighbors, predict(X_train) doesn't
    train_prediction = model.labels_
    test_prediction = model.predict(X_test)

    train_ba = balanced_accuracy_score(y_train, train_prediction)
    test_ba = balanced_accuracy_score(y_test, test_prediction)
    print("\nEx 2 - n_neighbors:", n_neighbors)
    print("Train BA:", round(train_ba, 4))
    print("Test BA:", round(test_ba, 4))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_labels(axes[0, 0], X_train, y_train, "Train - ground truth")
    plot_labels(axes[0, 1], X_train, train_prediction, "Train - predicted")
    plot_labels(axes[1, 0], X_test, y_test, "Test - ground truth")
    plot_labels(axes[1, 1], X_test, test_prediction, "Test - predicted")
    axes[0, 0].legend()
    fig.suptitle("Ex 2 - KNN, n_neighbors = " + str(n_neighbors))

# The anomalies here are 2 small clusters (around 20 points each in train). With
# n_neighbors = 1 the points of a cluster are each other's nearest neighbor, so
# their scores are small and some are missed (train BA 0.88), while some points
# on the edge of the inlier clusters get flagged. From 5 neighbors up the BA is
# 0.99 on train and 1.0 on test, even for 100 (more than a whole anomaly cluster).

plt.show()
