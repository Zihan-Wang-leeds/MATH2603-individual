
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(BASE_DIR, "..", "train.csv"))
val = pd.read_csv(os.path.join(BASE_DIR, "..", "val.csv"))

x_train = train[["x"]].values
y_train = train["y"].values
x_val = val[["x"]].values
y_val = val["y"].values

def polynomial_design_matrix(x):
    x = np.asarray(x).reshape(-1)
    return np.column_stack([np.ones(len(x)), x, x**2, x**3, x**4])

X_train_poly = polynomial_design_matrix(x_train)
X_val_poly = polynomial_design_matrix(x_val)

poly_coef, *_ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)
poly_train_pred = X_train_poly @ poly_coef
poly_val_pred = X_val_poly @ poly_coef

poly_train_mse = mean_squared_error(y_train, poly_train_pred)
poly_val_mse = mean_squared_error(y_val, poly_val_pred)

x_scaler = StandardScaler().fit(x_train)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

X_train_scaled = x_scaler.transform(x_train)
X_val_scaled = x_scaler.transform(x_val)
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

search_space = [
    {"hidden_layer_sizes": (16,), "alpha": 1e-5},
    {"hidden_layer_sizes": (32,), "alpha": 1e-5},
    {"hidden_layer_sizes": (64,), "alpha": 1e-5},
    {"hidden_layer_sizes": (32, 32), "alpha": 1e-5},
    {"hidden_layer_sizes": (64, 64), "alpha": 1e-5},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-5},
    {"hidden_layer_sizes": (64, 64), "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 64), "alpha": 1e-3},
]

nn_runs = []
for config in search_space:
    model = MLPRegressor(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=config["alpha"],
        learning_rate_init=0.01,
        max_iter=3000,
        random_state=42,
        early_stopping=False,
    )
    model.fit(X_train_scaled, y_train_scaled)

    train_pred = y_scaler.inverse_transform(model.predict(X_train_scaled).reshape(-1, 1)).ravel()
    val_pred = y_scaler.inverse_transform(model.predict(X_val_scaled).reshape(-1, 1)).ravel()

    nn_runs.append({
        "hidden_layer_sizes": config["hidden_layer_sizes"],
        "alpha": config["alpha"],
        "iterations": model.n_iter_,
        "train_mse": mean_squared_error(y_train, train_pred),
        "val_mse": mean_squared_error(y_val, val_pred),
        "model": model
    })

best_run = min(nn_runs, key=lambda d: d["val_mse"])
best_nn = best_run["model"]

nn_train_pred = y_scaler.inverse_transform(best_nn.predict(X_train_scaled).reshape(-1, 1)).ravel()
nn_val_pred = y_scaler.inverse_transform(best_nn.predict(X_val_scaled).reshape(-1, 1)).ravel()

nn_train_mse = mean_squared_error(y_train, nn_train_pred)
nn_val_mse = mean_squared_error(y_val, nn_val_pred)

x_grid = np.linspace(min(train["x"].min(), val["x"].min()), max(train["x"].max(), val["x"].max()), 500)
poly_grid_pred = polynomial_design_matrix(x_grid) @ poly_coef
nn_grid_pred = y_scaler.inverse_transform(best_nn.predict(x_scaler.transform(x_grid.reshape(-1, 1))).reshape(-1, 1)).ravel()

# Figures
plt.figure(figsize=(7, 4.5))
plt.scatter(train["x"], train["y"], s=18, alpha=0.65, label="Training data")
plt.scatter(val["x"], val["y"], s=22, alpha=0.75, label="Validation data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Noisy one-dimensional regression dataset")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dataset_scatter.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.scatter(train["x"], train["y"], s=15, alpha=0.45, label="Training data")
plt.scatter(val["x"], val["y"], s=20, alpha=0.55, label="Validation data")
plt.plot(x_grid, poly_grid_pred, linewidth=2.2, label="4th-order polynomial fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Least squares polynomial regression (validation MSE = {poly_val_mse:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "least_squares_fit.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.scatter(train["x"], train["y"], s=15, alpha=0.45, label="Training data")
plt.scatter(val["x"], val["y"], s=20, alpha=0.55, label="Validation data")
plt.plot(x_grid, nn_grid_pred, linewidth=2.2, label="Neural network fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Feedforward neural network (validation MSE = {nn_val_mse:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "neural_network_fit.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(best_nn.loss_curve_, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Training loss (scaled MSE)")
plt.title("Neural network optimisation trajectory")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "nn_loss_curve.png"), dpi=200)
plt.close()

results = {
    "polynomial_coefficients": {
        "a0": float(poly_coef[0]),
        "a1": float(poly_coef[1]),
        "a2": float(poly_coef[2]),
        "a3": float(poly_coef[3]),
        "a4": float(poly_coef[4]),
    },
    "polynomial_training_mse": float(poly_train_mse),
    "polynomial_validation_mse": float(poly_val_mse),
    "best_nn_hidden_layers": best_run["hidden_layer_sizes"],
    "best_nn_alpha": best_run["alpha"],
    "best_nn_iterations": best_run["iterations"],
    "best_nn_training_mse": float(nn_train_mse),
    "best_nn_validation_mse": float(nn_val_mse),
}

print("=== Polynomial baseline ===")
for key, value in results["polynomial_coefficients"].items():
    print(f"{key} = {value:.6f}")
print(f"Training MSE:   {poly_train_mse:.6f}")
print(f"Validation MSE: {poly_val_mse:.6f}")

print("\n=== Best neural network ===")
print("Hidden layers:", best_run["hidden_layer_sizes"])
print("Alpha:", best_run["alpha"])
print("Iterations:", best_run["iterations"])
print(f"Training MSE:   {nn_train_mse:.6f}")
print(f"Validation MSE: {nn_val_mse:.6f}")
