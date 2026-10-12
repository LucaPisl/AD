from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42,
)

print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)

print(X_train[:5])
print(y_train[:5])
