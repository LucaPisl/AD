import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.combination import average, maximization
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


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


# Ex 3
X, cluster = make_blobs(
    n_samples=[200, 100],
    centers=[(-10, -10), (10, 10)],
    cluster_std=[2, 6],
    random_state=42,
)

for n_neighbors in [5, 20, 50]:
    models = {
        "KNN": KNN(contamination=0.07, n_neighbors=n_neighbors),
        "LOF": LOF(contamination=0.07, n_neighbors=n_neighbors),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    print("\nEx 3 - n_neighbors:", n_neighbors)
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X)
        labels = model.labels_

        ax.scatter(X[labels == 0, 0], X[labels == 0, 1], s=15, color="blue")
        ax.scatter(X[labels == 1, 0], X[labels == 1, 1], s=15, color="red")
        ax.set_title(name)

        # how many outliers fall in each cluster (cluster 0 is the dense one)
        print(name, "outliers in dense cluster:", np.sum((labels == 1) & (cluster == 0)),
              "in sparse cluster:", np.sum((labels == 1) & (cluster == 1)))
    fig.suptitle("Ex 3 - n_neighbors = " + str(n_neighbors))

# KNN looks at the absolute distance to the neighbors, so all 21 outliers are
# in the sparse cluster, for any n_neighbors, and the dense cluster has none.
# LOF compares the density of a point with the density of its neighbors, so it
# also finds points on the edge of the dense cluster (13-16 of them) and fewer
# in the sparse cluster (5-8). The result doesn't change much with n_neighbors.


# Ex 4
data = loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()
contamination = np.mean(y)
print("\nEx 4 - contamination:", round(contamination, 4))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train, X_test = standardizer(X_train, X_test)

n_neighbors_list = np.linspace(30, 120, 10).astype(int)
train_scores = np.zeros((len(X_train), len(n_neighbors_list)))
test_scores = np.zeros((len(X_test), len(n_neighbors_list)))

for i, n_neighbors in enumerate(n_neighbors_list):
    model = KNN(contamination=contamination, n_neighbors=n_neighbors)
    model.fit(X_train)

    train_scores[:, i] = model.decision_scores_
    test_scores[:, i] = model.decision_function(X_test)

    train_ba = balanced_accuracy_score(y_train, model.labels_)
    test_ba = balanced_accuracy_score(y_test, model.predict(X_test))
    print("n_neighbors:", n_neighbors, "- train BA:", round(train_ba, 4),
          "test BA:", round(test_ba, 4))

# Normalize the scores of each model, using the train statistics
train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

strategies = {
    "Average": (average(train_scores_norm), average(test_scores_norm)),
    "Maximization": (maximization(train_scores_norm), maximization(test_scores_norm)),
}

for name, (train_final, test_final) in strategies.items():
    # the threshold comes from the train scores and is used for both sets
    threshold = np.quantile(train_final, 1 - contamination)
    train_prediction = (train_final > threshold).astype(int)
    test_prediction = (test_final > threshold).astype(int)

    print("\n" + name)
    print("Threshold:", round(threshold, 4))
    print("Train BA:", round(balanced_accuracy_score(y_train, train_prediction), 4))
    print("Test BA:", round(balanced_accuracy_score(y_test, test_prediction), 4))

# The BA of a single KNN goes up with n_neighbors (0.71 to 0.77 on train), so
# the ensembles are not better than the models with the biggest n_neighbors.
# Maximization is a bit better than average on test (0.805 vs 0.785).

plt.show()
