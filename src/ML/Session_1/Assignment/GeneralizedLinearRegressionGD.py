import numpy as np
import matplotlib.pyplot as plt


class GeneralizedLinearRegressionGD_Hazem:
    """
    Multiple Linear Regression using Gradient Descent
    Supports:
    - SSE & MSE tracking
    - Automatic reshaping (1D -> 2D)
    - Ridge (L2) regularization
    - Lasso (L1) regularization
    - Polynomial regression (feature transformation)
    """

    def __init__(self, alpha=0.01, num_iterations=100):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.w = None
        self.b = 0

        self.lambda_ = 0.0
        self.reg_type = "none"   # "none", "ridge", "lasso"

        self.SSE_values = []
        self.MSE_values = []

    # ---------- Utilities ----------

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def polynomial_transform(self, x, degree):
        x = np.asarray(x).reshape(-1, 1)
        return np.column_stack([x ** d for d in range(1, degree + 1)])

    # ---------- Core ML Methods ----------

    def initialize_parameters(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def predict(self, X):
        X = self._ensure_2d(X)
        return np.dot(X, self.w) + self.b

    def compute_gradients(self, X, y, y_hat):
        n = len(y)

        dw = (1 / n) * np.dot(X.T, (y_hat - y))
        db = (1 / n) * np.sum(y_hat - y)

        # Regularization
        if self.reg_type == "ridge":          # L2
            dw += self.lambda_ * self.w

        elif self.reg_type == "lasso":        # L1
            dw += self.lambda_ * np.sign(self.w)

        return dw, db

    def compute_sse(self, y, y_hat):
        return np.sum((y_hat - y) ** 2)

    def compute_mse(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)

    # ---------- Training ----------

    def fit(self, X, y):
        X = self._ensure_2d(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)

        self.SSE_values = []
        self.MSE_values = []

        for i in range(self.num_iterations):

            y_hat = self.predict(X)

            dw, db = self.compute_gradients(X, y, y_hat)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

            # re-compute predictions
            y_hat = self.predict(X)

            sse = self.compute_sse(y, y_hat)
            mse = self.compute_mse(y, y_hat)

            self.SSE_values.append(sse)
            self.MSE_values.append(mse)

            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}, SSE: {sse}, MSE: {mse}")

        return self

    # ---------- Explicit APIs ----------

    def fit_linear(self, X, y):
        self.reg_type = "none"
        return self.fit(X, y)

    def fit_ridge(self, X, y, lambda_):
        self.lambda_ = lambda_
        self.reg_type = "ridge"
        return self.fit(X, y)

    def fit_lasso(self, X, y, lambda_):
        self.lambda_ = lambda_
        self.reg_type = "lasso"
        return self.fit(X, y)

    def fit_polynomial(self, x, y, degree):
        X_poly = self.polynomial_transform(x, degree)
        self.reg_type = "none"
        return self.fit(X_poly, y)

    def fit_polynomial_ridge(self, x, y, degree, lambda_):
        X_poly = self.polynomial_transform(x, degree)
        self.lambda_ = lambda_
        self.reg_type = "ridge"
        return self.fit(X_poly, y)

    def fit_polynomial_lasso(self, x, y, degree, lambda_):
        X_poly = self.polynomial_transform(x, degree)
        self.lambda_ = lambda_
        self.reg_type = "lasso"
        return self.fit(X_poly, y)

    # ---------- Visualization ----------

    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.MSE_values, label="MSE")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title("MSE over Iterations")
        plt.legend()
        plt.show()

    def get_params(self):
        return {
            "weights": self.w,
            "bias": self.b
        }